import torch
import os
from diffusers import WanVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# 1. 强制系统级优化：防止显存碎片导致的数值计算错误
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_stable_wan():
    # 建议统一使用 Hugging Face 的路径格式，diffusers 兼容性更好
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    
    print(f"正在启动全精度（Float32）模式...")

    # 2. 核心修改：强制 VAE 使用 float32
    # 噪声通常源于 VAE 在 bf16 下解码失败
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )

    # 3. 核心修改：强制 Pipeline 全程使用 float32
    # fp32 虽然慢且占显存，但它能从数学上杜绝 NaN（空值）和噪声产生
    pipe = WanVideoPipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=torch.float32
    )

    # 4. 显存管理：即便有 24GB/48GB 显存，跑 fp32 也必须开启卸载
    # 否则 Text Encoder 和 Transformer 同时存在时极易爆显存
    pipe.enable_model_cpu_offload()

    # 5. 关键救命药：开启 VAE 的空间与时间分块
    # 解决 81 帧视频在解码阶段因数值堆叠导致的雪花屏
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # 6. 精简提示词（排除冗长负面词可能带来的干扰）
    prompt = "A cinematic shot of a cute cat walking on the green grass, realistic, high quality, 4k."
    negative_prompt = "low quality, blurry, static, noisy, gray, monochrome, distorted"

    print("开始生成采样 (Float32推理)...")
    
    with torch.inference_mode():
        # 建议先跑 41 帧验证稳定性，成功后再改回 81
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=41,          
            guidance_scale=5.0,     
            num_inference_steps=25, # fp32 下 25 步已经足够清晰
        ).frames[0]

    # 7. 保存结果
    output_path = "stable_output.mp4"
    export_to_video(output, output_path, fps=15)
    print(f"✅ 任务完成！请检查当前目录下的 {output_path}")

if __name__ == "__main__":
    run_stable_wan()