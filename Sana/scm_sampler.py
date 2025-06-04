"""
SCM (Smart Consistency Model) Sampler for ComfyUI
Based on ComfyUI's LCM implementation

Usage:
1. Connect ScmModelSampling node to your model, set cfg_scale
2. Set KSampler cfg=1.0 (disable KSampler's CFG)
3. Use "scm" sampler
"""

import torch
import comfy.samplers
import comfy.model_sampling
from tqdm.auto import trange


# SCM sampling function
@torch.no_grad()
def sample_scm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """SCM sampling algorithm"""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    
    model_options = extra_args.get("model_options", {})
    cfg_scale = model_options.get("cfg_scale", 4.5)
    
    if "transformer_options" not in model_options:
        model_options["transformer_options"] = {}
    model_options["transformer_options"]["cfg_scale"] = cfg_scale
    
    # Update extra_args
    extra_args['model_options'] = model_options
    
    num_inference_steps = len(sigmas) - 1
    max_timesteps = 1.57080
    sigma_data = 0.5
    
    # Generate timestep sequence
    if num_inference_steps == 2:
        # Special case: 2 steps use [1.57080, 1.3, 0]
        timesteps = torch.tensor([max_timesteps, 1.3, 0.0], device=x.device, dtype=x.dtype)
    else:
        # General case: use linspace
        timesteps = torch.linspace(max_timesteps, 0, num_inference_steps + 1, device=x.device).float()
    
    print(f"SCM timesteps: {timesteps}")
    
    latents = x * sigma_data
    denoised = None
    
    # SCM MultiStep Sampling Loop
    for i, t in enumerate(timesteps[:-1]):
        timestep = t.expand(latents.shape[0])
        
        model_pred = sigma_data * model(
            latents / sigma_data,
            timestep,
            **extra_args
        )

        if callback is not None:
            callback({'x': latents, 'i': i, 'sigma': t, 'sigma_hat': t, 'denoised': model_pred})

        s = t
        next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0, device=t.device)
        
        # SCM denoising: pred_x0 = cos(s) * x - sin(s) * model_output
        denoised = torch.cos(s) * latents - torch.sin(s) * model_pred
        
        if next_t > 0:
            noise = noise_sampler(s, next_t) * sigma_data
            latents = torch.cos(next_t) * denoised + torch.sin(next_t) * noise
        else:
            latents = denoised
    
    denoised = denoised / sigma_data

    return denoised if denoised is not None else latents


class SCM(comfy.model_sampling.EPS):
    """SCM Model Sampling class"""
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        
        return model_output


class ScmModelSamplingDiscreteDistilled(comfy.model_sampling.ModelSamplingDiscrete):
    """SCM distilled sampling"""
    
    original_timesteps = 50 

    def __init__(self, model_config=None, zsnr=None):
        super().__init__(model_config, zsnr=zsnr)

        self.skip_steps = self.num_timesteps // self.original_timesteps

        sigmas_valid = torch.zeros((self.original_timesteps), dtype=torch.float32)
        for x in range(self.original_timesteps):
            sigmas_valid[self.original_timesteps - 1 - x] = self.sigmas[self.num_timesteps - 1 - x * self.skip_steps]

        self.set_sigmas(sigmas_valid)

    def timestep(self, sigma):
        # SCM: directly return sigma as timestep, no conversion
        # Because in SCM sigma is our timestep (0 to 1.57080 range)
        return sigma

    def sigma(self, timestep):
        # SCM: directly return timestep as sigma for consistency
        return timestep
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # SCM: no additional noise scaling, return noise directly
        return noise
    
    def inverse_noise_scaling(self, sigma, latent):
        # SCM: no inverse scaling, sample_scm already returns correct result
        return latent


# ComfyUI node class
class ScmModelSampling:
    """SCM Model Sampling node - supports CFG scale passing"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG scale for SCM model, passed directly to model"}),
                "zsnr": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "ExtraModels/Sana"
    TITLE = "SCM Model Sampling"

    def patch(self, model, cfg_scale, zsnr=False):
        m = model.clone()

        # Create SCM sampling class
        class ScmModelSamplingAdvanced(ScmModelSamplingDiscreteDistilled, SCM):
            pass

        model_sampling = ScmModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)
        m.add_object_patch("model_sampling", model_sampling)
        
        # Store CFG scale to model_options
        if not hasattr(m, 'model_options'):
            m.model_options = {}
        m.model_options['cfg_scale'] = cfg_scale
        
        print(f"SCM Model Sampling: CFG scale set to {cfg_scale}")
        return (m,)


# Register SCM sampler to ComfyUI
def register_scm_sampler():
    """Register SCM sampler"""
    # Register to KSAMPLER_NAMES
    if "scm" not in comfy.samplers.KSAMPLER_NAMES:
        comfy.samplers.KSAMPLER_NAMES.append("scm")
    
    # Update SAMPLER_NAMES
    if "scm" not in comfy.samplers.SAMPLER_NAMES:
        comfy.samplers.SAMPLER_NAMES.append("scm")
    
    # Register sampling function to k_diffusion module
    if not hasattr(comfy.k_diffusion.sampling, "sample_scm"):
        setattr(comfy.k_diffusion.sampling, "sample_scm", sample_scm)


# Auto register
register_scm_sampler()


# Debug function
def check_scm_registration():
    """Check SCM sampler registration status"""
    status = {
        "scm_in_KSAMPLER_NAMES": "scm" in comfy.samplers.KSAMPLER_NAMES,
        "scm_in_SAMPLER_NAMES": "scm" in comfy.samplers.SAMPLER_NAMES,
        "sample_scm_function_exists": hasattr(comfy.k_diffusion.sampling, "sample_scm"),
        "KSAMPLER_NAMES": comfy.samplers.KSAMPLER_NAMES,
        "SAMPLER_NAMES": comfy.samplers.SAMPLER_NAMES,
    }
    print("SCM Registration Status:", status)
    return status


# Check status on import
check_scm_registration()


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ScmModelSampling": ScmModelSampling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScmModelSampling": "SCM Model Sampling",
}

