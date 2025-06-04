"""
SCM (Smart Consistency Model) Sampler for ComfyUI
基于ComfyUI的LCM实现改写的简单SCM版本
"""

import torch
import comfy.samplers
import comfy.model_sampling
from tqdm.auto import trange


# SCM采样函数 - 基于用户提供的时间步定义
@torch.no_grad()
def sample_scm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    """SCM采样算法 - 使用直接定义的时间步序列"""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    
    # SCM时间步定义：从1.57080到0的线性序列
    num_inference_steps = len(sigmas) - 1  # 仍需要从sigmas获取步数信息
    max_timesteps = 1.57080
    sigma_data = 0.5  # 根据你的实现，sigma_data通常是0.5
    
    # 生成时间步序列
    if num_inference_steps == 2:
        # 特殊情况：2步使用 [1.57080, 1.3, 0]
        timesteps = torch.tensor([max_timesteps, 1.3, 0.0], device=x.device, dtype=x.dtype)
    else:
        # 一般情况：使用linspace
        timesteps = torch.linspace(max_timesteps, 0, num_inference_steps + 1, device=x.device).float()
    
    print(f"SCM timesteps: {timesteps}")
    
    latents = x * sigma_data
    denoised = None
    extra_args["model_options"]["cfg_scale"] = 4.5
    
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
        # SCM: 直接返回sigma作为时间步，不做转换
        # 因为在SCM中sigma就是我们的时间步（0到1.57080范围）
        return sigma

    def sigma(self, timestep):
        # SCM: 直接返回timestep作为sigma，保持一致性
        return timestep
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # SCM: 不进行额外的noise scaling，直接返回noise
        return noise
    
    def inverse_noise_scaling(self, sigma, latent):
        # SCM: 不进行inverse scaling，因为我们的sample_scm已经返回了正确的结果
        return latent


# ComfyUI节点类
class ScmModelSampling:
    """SCM Model Sampling节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "zsnr": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "ExtraModels/Sana"
    TITLE = "SCM Model Sampling"

    def patch(self, model, zsnr=False):
        m = model.clone()

        # 创建SCM采样类
        class ScmModelSamplingAdvanced(ScmModelSamplingDiscreteDistilled, SCM):
            pass

        model_sampling = ScmModelSamplingAdvanced(model.model.model_config, zsnr=zsnr)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)


# 注册SCM采样器到ComfyUI
def register_scm_sampler():
    """注册SCM采样器"""
    # 注册到KSAMPLER_NAMES
    if "scm" not in comfy.samplers.KSAMPLER_NAMES:
        comfy.samplers.KSAMPLER_NAMES.append("scm")
    
    # 更新SAMPLER_NAMES
    if "scm" not in comfy.samplers.SAMPLER_NAMES:
        comfy.samplers.SAMPLER_NAMES.append("scm")
    
    # 注册采样函数到k_diffusion模块
    if not hasattr(comfy.k_diffusion.sampling, "sample_scm"):
        setattr(comfy.k_diffusion.sampling, "sample_scm", sample_scm)


# 自动注册
register_scm_sampler()


# 调试函数
def check_scm_registration():
    """检查SCM采样器的注册状态"""
    status = {
        "scm_in_KSAMPLER_NAMES": "scm" in comfy.samplers.KSAMPLER_NAMES,
        "scm_in_SAMPLER_NAMES": "scm" in comfy.samplers.SAMPLER_NAMES,
        "sample_scm_function_exists": hasattr(comfy.k_diffusion.sampling, "sample_scm"),
        "KSAMPLER_NAMES": comfy.samplers.KSAMPLER_NAMES,
        "SAMPLER_NAMES": comfy.samplers.SAMPLER_NAMES,
    }
    print("SCM Registration Status:", status)
    return status


# 在导入时检查状态
check_scm_registration()


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ScmModelSampling": ScmModelSampling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScmModelSampling": "SCM Model Sampling",
}

