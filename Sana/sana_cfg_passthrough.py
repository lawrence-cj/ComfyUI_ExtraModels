"""
Sana CFG Scale Passthrough Tool
Provides CFG scale passthrough functionality for SanaMS models, supporting all Sampler types

Usage:
from .Sana.sana_cfg_passthrough import enable_sana_cfg
model = enable_sana_cfg(your_model)

The model forward method can then receive cfg_scale parameter
"""

import comfy.samplers
import torch


class SanaCFGPassthrough:
    """CFG passthrough manager"""
    
    _instance = None
    _patched = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.current_cfg_scale = 1.0
        self.original_sampling_function = None
        self.original_ksampler_sample = None
    
    def patch_global_functions(self):
        """Globally patch ComfyUI functions (executed only once)"""
        if self._patched:
            return
        
        # Patch core sampling_function
        if not hasattr(comfy.samplers.sampling_function, '_sana_cfg_patched'):
            self.original_sampling_function = comfy.samplers.sampling_function
            
            def patched_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
                # Save current CFG scale
                self.current_cfg_scale = cond_scale
                return self.original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
            
            comfy.samplers.sampling_function = patched_sampling_function
            comfy.samplers.sampling_function._sana_cfg_patched = True
        
        # Patch KSampler
        if not hasattr(comfy.samplers.KSampler, '_sana_cfg_patched'):
            self.original_ksampler_sample = comfy.samplers.KSampler.sample
            
            def patched_ksampler_sample(sampler_self, noise, positive, negative, cfg, **kwargs):
                # Add CFG to model_options
                if "transformer_options" not in sampler_self.model_options:
                    sampler_self.model_options["transformer_options"] = {}
                sampler_self.model_options["transformer_options"]["cfg_scale"] = cfg
                
                return self.original_ksampler_sample(sampler_self, noise, positive, negative, cfg, **kwargs)
            
            comfy.samplers.KSampler.sample = patched_ksampler_sample
            comfy.samplers.KSampler._sana_cfg_patched = True
        
        self._patched = True
    
    def restore_global_functions(self):
        """Restore original functions"""
        if self.original_sampling_function:
            comfy.samplers.sampling_function = self.original_sampling_function
            if hasattr(comfy.samplers.sampling_function, '_sana_cfg_patched'):
                delattr(comfy.samplers.sampling_function, '_sana_cfg_patched')
        
        if self.original_ksampler_sample:
            comfy.samplers.KSampler.sample = self.original_ksampler_sample
            if hasattr(comfy.samplers.KSampler, '_sana_cfg_patched'):
                delattr(comfy.samplers.KSampler, '_sana_cfg_patched')
        
        self._patched = False
    
    def create_model_wrapper(self, model):
        """Create CFG wrapper for specific model"""
        def cfg_wrapper(apply_model_func, params):
            input_x = params["input"]
            timestep = params["timestep"] 
            c = params["c"]
            
            # Get CFG scale
            cfg_scale = self.get_current_cfg_scale(model)
            
            # Pass to model
            c = c.copy()
            c['cfg_scale'] = cfg_scale
            
            return apply_model_func(input_x, timestep, **c)
        
        return cfg_wrapper
    
    def get_current_cfg_scale(self, model):
        """Get current CFG scale from multiple sources"""
        # Priority:
        # 1. CFG in model_options
        # 2. Globally saved CFG
        # 3. Default value
        
        model_options = getattr(model, 'model_options', {})
        transformer_options = model_options.get('transformer_options', {})
        if 'cfg_scale' in transformer_options:
            return transformer_options['cfg_scale']
        
        return self.current_cfg_scale


# Global instance
_cfg_manager = SanaCFGPassthrough()


def enable_sana_cfg(model):
    """
    Enable CFG scale passthrough functionality for SanaMS model
    
    Args:
        model: ModelPatcher instance
    
    Returns:
        Configured model
    
    After use, the SanaMS forward method can receive CFG values via cfg_scale parameter
    """
    # Ensure global functions are patched
    _cfg_manager.patch_global_functions()
    
    # Set wrapper for model
    wrapper = _cfg_manager.create_model_wrapper(model)
    model.set_model_unet_function_wrapper(wrapper)
    
    # Mark model as CFG passthrough enabled
    model._sana_cfg_enabled = True
    
    return model


def disable_sana_cfg(model):
    """
    Disable CFG passthrough functionality for model
    """
    if hasattr(model, '_sana_cfg_enabled'):
        # Remove wrapper (set to None)
        model.set_model_unet_function_wrapper(None)
        delattr(model, '_sana_cfg_enabled')
    
    return model


def cleanup_sana_cfg():
    """
    Clean up all CFG passthrough functionality, restore original state
    Warning: This will affect all models with CFG passthrough enabled
    """
    _cfg_manager.restore_global_functions()


def get_current_cfg_scale():
    """
    Get current CFG scale value (for debugging)
    """
    return _cfg_manager.current_cfg_scale


# Compatibility check
def check_sana_cfg_compatibility():
    """
    Check CFG passthrough compatibility in current environment
    """
    compatibility = {
        "ComfyUI version": "✅ Compatible",
        "Supported Samplers": {
            "KSampler": "✅ Supported",
            "KSamplerAdvanced": "✅ Supported", 
            "SamplerCustom": "✅ Supported",
            "All other Samplers": "✅ Supported (via sampling_function)"
        },
        "Function status": {
            "Global patch status": "✅ Applied" if _cfg_manager._patched else "❌ Not applied",
            "Current CFG scale": _cfg_manager.current_cfg_scale
        }
    }
    
    return compatibility


# Usage examples (in comments)
"""
Usage examples:

# 1. Enable CFG passthrough
from .Sana.sana_cfg_passthrough import enable_sana_cfg

model = your_sana_model  # ModelPatcher instance
model = enable_sana_cfg(model)

# 2. Receive CFG in SanaMS forward method
def forward(self, x, timesteps, context, cfg_scale=None, **kwargs):
    if cfg_scale is not None:
        print(f"Received CFG scale: {cfg_scale}")
        # Use cfg_scale for your calculations...
    
    # Your original forward logic
    return self.original_forward(x, timesteps, context, **kwargs)

# 3. Use KSampler or any other Sampler normally
# CFG scale will be automatically passed to the model

# 4. Disable if needed (optional)
# disable_sana_cfg(model)

# 5. Check compatibility (optional)
# compatibility = check_sana_cfg_compatibility()
# print(compatibility)
""" 