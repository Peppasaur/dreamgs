# Wan Utils SDS Implementation - Fixes Applied

## Issues Found and Fixed

### 1. **VAE Encoding/Decoding (CRITICAL)**
**Problem**: Original code used `scaling_factor` normalization (standard SD approach)
```python
# WRONG
latents = posterior.sample() * self.vae.config.scaling_factor
```

**Fix**: AutoencoderKLWan uses `latents_mean` and `latents_std` normalization
```python
# CORRECT
latents = (latents - latents_mean) * latents_std
```

**Impact**: Without proper normalization, the latent space representation would be completely wrong, leading to meaningless gradients in SDS.

---

### 2. **Latent Shape (CRITICAL)**
**Problem**: Treated Wan as image model with 4D tensors `[B, C, H, W]`

**Fix**: Wan is a video model requiring 5D tensors `[B, C, T, H, W]`
- For single images, use T=1: `[B, C, 1, H, W]`
- Added `unsqueeze(2)` when encoding and `squeeze(2)` when decoding

**Impact**: Incompatible tensor shapes would cause runtime errors or incorrect model behavior.

---

### 3. **Transformer API (CRITICAL)**
**Problem**: Called transformer with positional argument
```python
# WRONG
velocity_pred = self.transformer(
    latent_model_input,  # positional
    timestep=tt,
    ...
)
```

**Fix**: Use named parameter `hidden_states=`
```python
# CORRECT
velocity_pred = self.transformer(
    hidden_states=latent_model_input,
    timestep=timestep,
    ...
)
```

**Impact**: API mismatch could cause errors or use wrong model inputs.

---

### 4. **Classifier-Free Guidance (CRITICAL)**
**Problem**: Concatenated inputs and called transformer once
```python
# WRONG
latent_model_input = torch.cat([latents_noisy] * 2)
embeddings = self.embeddings.repeat(batch_size, 1, 1)  # [neg, pos] concatenated
velocity_pred = self.transformer(...)
```

**Fix**: Call transformer twice separately with cache context
```python
# CORRECT
with self.transformer.cache_context("cond"):
    velocity_pred_pos = self.transformer(
        encoder_hidden_states=self.embeddings[1:].repeat(batch_size, 1, 1)  # positive only
    )

with self.transformer.cache_context("uncond"):
    velocity_pred_uncond = self.transformer(
        encoder_hidden_states=self.embeddings[:1].repeat(batch_size, 1, 1)  # negative only
    )

velocity_pred = velocity_pred_uncond + guidance_scale * (velocity_pred_pos - velocity_pred_uncond)
```

**Impact**: Incorrect CFG implementation leads to wrong gradient directions and poor SDS performance.

---

### 5. **Timestep Expansion**
**Problem**: Used concatenated timesteps `torch.cat([t] * 2)`

**Fix**: Properly expand to batch size
```python
# CORRECT
timestep = t.expand(batch_size)
```

**Impact**: Ensures correct timestep input format for the model.

---

### 6. **Data Types**
**Problem**: Mixed dtypes without proper conversion

**Fix**: 
- Use `torch.bfloat16` for transformer (better than fp16)
- Use `torch.float32` for VAE (as recommended)
- Convert latents to transformer dtype before feeding: `latents.to(self.transformer.dtype)`

**Impact**: Prevents numerical instability and ensures compatibility.

---

### 7. **Tau Expansion for 5D Tensors**
**Problem**: Expanded tau for 4D: `tau.view(batch_size, 1, 1, 1)`

**Fix**: Expand for 5D: `tau.view(batch_size, 1, 1, 1, 1)`

**Impact**: Correct broadcasting for noisy latent computation.

---

## Verification Checklist

✅ VAE encoding uses Wan-specific normalization (`latents_mean`, `latents_std`)
✅ VAE decoding applies inverse normalization correctly  
✅ All latent tensors are 5D: `[B, C, 1, H, W]`
✅ Transformer called with `hidden_states=` parameter
✅ Classifier-free guidance uses separate transformer calls with cache context
✅ Timestep properly expanded to batch size
✅ Proper dtype handling (bfloat16 for transformer, float32 for VAE)
✅ RFSDS formula correctly implemented: `grad = velocity_pred - noise + latents`
✅ Weighted noise sampling with annealing schedule

---

## RFSDS Algorithm Summary

The implementation correctly follows the paper's RFSDS algorithm:

1. **Encode image to latents**: `z = VAE.encode(x)`
2. **Sample noise level**: `τ ~ w(τ) = τ²` with annealing `h(τᵢ) = 1 - i/(I+1)`
3. **Create noisy latent**: `z_τ = (1-τ)z + τϵ`
4. **Predict velocity**: `v̂ = TransformerRF(z_τ, τ, y)` with CFG
5. **Compute gradient**: `∇L = v̂ - ϵ + z` (no weighting term)
6. **Apply gradient trick**: `target = (z - ∇L).detach()`, `loss = MSE(z, target)`

This implementation is now compatible with the official Wan pipeline architecture.
