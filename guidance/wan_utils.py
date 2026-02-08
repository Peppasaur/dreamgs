from transformers import logging
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class WanRectifiedFlow(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=True,
        model_key="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.model_key = model_key

        # Use bfloat16 for better stability with Wan models
        self.dtype = torch.bfloat16 if fp16 else torch.float32

        print(f"[INFO] Loading Wan 2.2 Rectified Flow model from {model_key}...")
        
        local_model_path = "/home/featurize/Wan2.2-TI2V-5B-Diffusers"
        
        self.vae = AutoencoderKLWan.from_pretrained(
            local_model_path,       # 替换原先的 repo_id (model_key)
            subfolder="vae",        # 依然需要 subfolder，因为 VAE 在本地目录的 vae 子文件夹里
            torch_dtype=self.dtype,
            local_files_only=True   # 强制只检查本地，防止因为网络问题报错
        )
        
        # Load complete pipeline
        self.pipe = WanPipeline.from_pretrained(
            local_model_path,
            vae=self.vae,
            torch_dtype=self.dtype,
            local_files_only=True
        )
        
        # Handle model offloading and memory optimization
        self.t5_on_cpu = False
        
        if offload_model or vram_O:
            # Enable model CPU offload for memory efficiency
            self.pipe.enable_model_cpu_offload()
            print(f"[INFO] Enabled model CPU offload")
            
            # Enable VAE slicing and tiling
            if hasattr(self.vae, 'enable_slicing'):
                self.vae.enable_slicing()
            if hasattr(self.vae, 'enable_tiling'):
                self.vae.enable_tiling()
            
            # Mark that T5 will be managed by CPU offload
            self.t5_on_cpu = True
        else:
            self.pipe.to(device)

        # Access components from pipeline
        self.transformer = self.pipe.transformer
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler

        # Handle explicit T5 CPU offloading
        if t5_cpu and not self.t5_on_cpu:
            print(f"[INFO] Moving T5 text encoder to CPU")
            self.text_encoder.to("cpu")
            self.t5_on_cpu = True

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.embeddings = None
        
        # For weighted noise sampling in RFSDS
        self.current_iter = 0
        self.total_iters = 1000  # Will be set from training loop

        self.alphas = self.scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        """
        Encode text prompts using the pipeline's encode_prompt method.
        Uses T5 text encoder with proper preprocessing.
        """
        # Temporarily move T5 to GPU if it's on CPU
        if self.t5_on_cpu:
            self.text_encoder.to(self.device)
        
        # Use the pipeline's encode_prompt function
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompts,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Move T5 back to CPU if needed
        if self.t5_on_cpu:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()  # Free GPU memory
        
        # Store in format [negative, positive] for compatibility
        self.embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W] -> latents: [B, C, 1, H//8, W//8]
        # Wan VAE expects 5D input for video: [B, C, T, H, W]
        imgs = 2 * imgs - 1
        
        # Add temporal dimension: [B, 3, H, W] -> [B, 3, 1, H, W]
        imgs = imgs.unsqueeze(2)
        
        # Encode to latents
        latents = self.vae.encode(imgs).latent_dist.sample()
        
        # Apply Wan-specific normalization
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        
        return latents

    def decode_latents(self, latents):
        # latents: [B, C, 1, H, W] -> imgs: [B, 3, H, W]
        
        # Apply inverse Wan-specific normalization
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        
        # Decode from latents
        imgs = self.vae.decode(latents).sample
        
        # Remove temporal dimension: [B, 3, 1, H, W] -> [B, 3, H, W]
        imgs = imgs.squeeze(2)
        
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def sample_noise_level(self, batch_size):
        """
        Sample noise level τ using weighted distribution w(τ) = τ²/∫τ²dτ
        with annealing schedule h(τ_i) = 1 - i/(I+1)
        
        From paper equation (4) and section 3.2
        """
        # Annealing schedule: gradually decrease noise level over training
        h_tau = 1 - self.current_iter / (self.total_iters + 1)
        h_tau = max(0.02, min(0.98, h_tau))  # Clamp to valid range
        
        # Sample from weighted distribution w(τ) = τ²
        # Using inverse CDF: τ = τ_i^(1/3) where τ_i ~ U(0, h_tau³)
        u = torch.rand(batch_size, device=self.device)
        tau = (u * (h_tau ** 3)) ** (1/3)
        
        return tau

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
    ):
        """
        Rectified Flow SDS (RFSDS) training step
        Implements equation (3) from the paper:
        ∇_θ L_W-RFSDS(θ; z, y) = E_τ~ŵ(τ),ϵ [(v̂(z_τ; τ, y) - ϵ + z) ∂z/∂θ]
        
        Where:
        - v̂ is the predicted velocity from rectified flow model
        - z is the clean latent (rendered image)
        - ϵ is the added noise
        - z_τ = (1-τ)z + τϵ is the noisy latent
        """
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
            # Add temporal dimension for Wan: [B, C, H, W] -> [B, C, 1, H, W]
            latents = latents.unsqueeze(2)
        else:
            # Encode image into latents with VAE
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)  # Returns [B, C, 1, H, W]

        # Ensure latents are in float32 for gradient computation
        latents = latents.float()
        
        # Sample noise level τ using weighted distribution
        if step_ratio is not None:
            # Use step_ratio to set current iteration for annealing
            self.current_iter = int(step_ratio * self.total_iters)
        
        #tau = self.sample_noise_level(batch_size)
        tau = torch.full((batch_size,), 1 - step_ratio, dtype=torch.float16, device=self.device)
        
        # Convert τ to timestep for the scheduler
        # Rectified flow: τ ∈ [0, 1] maps to timesteps
        t = (tau * self.num_train_timesteps).long().clamp(self.min_step, self.max_step)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # Sample noise ϵ with same shape as latents [B, C, 1, H, W]
        noise = torch.randn_like(latents)
        
        # Create noisy latent: z_τ = (1-τ)z + τϵ
        tau_expanded = tau.view(batch_size, 1, 1, 1, 1).float()  # Expand for 5D and ensure float32
        latents_noisy = (1 - tau_expanded) * latents + tau_expanded * noise

        # Predict velocity with transformer, NO grad!
        with torch.no_grad():
            # Convert to transformer dtype
            latent_model_input = latents_noisy.to(self.transformer.dtype)
            
            # Expand timestep to batch size
            timestep = t.expand(batch_size)
            
            # For rectified flow, the model predicts velocity v̂
            # Call transformer with conditional embeddings (positive prompt)
            with self.transformer.cache_context("cond"):
                velocity_pred_pos = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=self.embeddings[1:].repeat(batch_size, 1, 1),  # positive embeddings
                    return_dict=False,
                )[0]

            # Perform classifier-free guidance
            if guidance_scale > 1.0:
                # Call transformer with unconditional embeddings (negative prompt)
                with self.transformer.cache_context("uncond"):
                    velocity_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=self.embeddings[:1].repeat(batch_size, 1, 1),  # negative embeddings
                        return_dict=False,
                    )[0]
                
                velocity_pred = velocity_pred_uncond + guidance_scale * (
                    velocity_pred_pos - velocity_pred_uncond
                )
            else:
                velocity_pred = velocity_pred_pos

        # Compute RFSDS gradient: v̂(z_τ; τ, y) - ϵ + z
        # Note: The weighting term is eliminated as per equation (3)
        # Convert velocity_pred back to float32 for gradient computation
        velocity_pred = velocity_pred.float()
        
        grad = w*(velocity_pred - noise + latents)
        grad = torch.nan_to_num(grad)
        print("grad",torch.mean(grad))
        print("noise",torch.mean(noise))
        print("latents",torch.mean(latents))
        print("velocity_pred",torch.mean(velocity_pred))
        print("w",torch.mean(w))
        print("velocity_pred_uncond",torch.mean(velocity_pred_uncond))
        print("velocity_pred_pos",torch.mean(velocity_pred_pos))
        # Create target and compute loss using gradient trick
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction='sum') / batch_size

        # Ensure loss is in float32 for compatibility
        return loss.float()

    @torch.no_grad()
    def refine(
        self, 
        pred_rgb,
        guidance_scale=7.5, 
        steps=50, 
        strength=0.8,
    ):
        """
        Refine rendered images using rectified flow model
        """
        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))  # [B, C, 1, H, W]

        self.scheduler.set_timesteps(steps, device=self.device)
        init_step = int(steps * strength)
        
        # Add noise for denoising
        if init_step > 0:
            noise = torch.randn_like(latents)
            timesteps = self.scheduler.timesteps[init_step]
            latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            latent_model_input = latents.to(self.transformer.dtype)
            timestep = t.expand(batch_size)

            # Conditional prediction
            with self.transformer.cache_context("cond"):
                velocity_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=self.embeddings[1:].repeat(batch_size, 1, 1),
                    return_dict=False,
                )[0]

            # Classifier-free guidance
            if guidance_scale > 1.0:
                with self.transformer.cache_context("uncond"):
                    velocity_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=self.embeddings[:1].repeat(batch_size, 1, 1),
                        return_dict=False,
                    )[0]
                
                velocity_pred = velocity_pred_uncond + guidance_scale * (
                    velocity_pred_cond - velocity_pred_uncond
                )
            else:
                velocity_pred = velocity_pred_cond
            
            latents = self.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        imgs = self.decode_latents(latents)
        return imgs

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        batch_size = self.embeddings.shape[0] // 2
        
        if latents is None:
            # Wan uses 16 latent channels and 5D shape [B, C, T, H, W]
            num_channels = self.transformer.config.in_channels
            latents = torch.randn(
                (
                    batch_size,
                    num_channels,
                    1,  # single frame
                    height // 8,
                    width // 8,
                ),
                device=self.device,
                dtype=torch.float32,  # Latents are generated in float32
            )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = latents.to(self.transformer.dtype)
            timestep = t.expand(batch_size)
            
            # Conditional prediction
            with self.transformer.cache_context("cond"):
                velocity_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=self.embeddings[1:].repeat(batch_size, 1, 1),
                    return_dict=False,
                )[0]

            # Classifier-free guidance
            if guidance_scale > 1.0:
                with self.transformer.cache_context("uncond"):
                    velocity_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=self.embeddings[:1].repeat(batch_size, 1, 1),
                        return_dict=False,
                    )[0]
                
                velocity_pred = velocity_pred_uncond + guidance_scale * (
                    velocity_pred_cond - velocity_pred_uncond
                )
            else:
                velocity_pred = velocity_pred_cond

            latents = self.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Img latents -> imgs
        imgs = self.decode_latents(latents)

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--model_key",
        type=str,
        default="Wan-AI/wan2.2",
        help="Hugging Face model key for Wan 2.2",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    wan = WanRectifiedFlow(device, opt.fp16, opt.vram_O, opt.model_key)

    imgs = wan.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
