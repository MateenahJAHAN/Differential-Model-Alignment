"""
Model loading and management with LoRA-aware optimization.
"""

import torch
from torch import nn
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import warnings

from .base import ModelWrapper
from .config import ModelConfig

logger = logging.getLogger(__name__)


class HuggingFaceModelWrapper(ModelWrapper):
    """Wrapper for HuggingFace transformers models with LoRA support."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = None
        self.activation_cache = {}
        self.hooks = []
        
    def load_model(self):
        """Load model with appropriate configuration and optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Prepare quantization config if needed
        quantization_config = None
        if self.config.load_in_8bit or self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        # Load base model
        model_kwargs = {
            "torch_dtype": getattr(torch, self.config.dtype),
            "device_map": "auto" if self.config.device == "cuda" else None,
            "trust_remote_code": True,
            "quantization_config": quantization_config,
        }
        
        # Use flash attention if available
        if self.config.use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
            
        # Load from local path or HuggingFace hub
        model_path = self.config.model_path or self.config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs
        )
        
        # Load LoRA adapter if specified
        if self.config.is_lora and self.config.lora_path:
            logger.info(f"Loading LoRA adapter from: {self.config.lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.lora_path,
                torch_dtype=getattr(torch, self.config.dtype),
            )
            # Set adapter scaling
            self.model.base_model.model.lora_scaling = self.config.lora_alpha
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Move to device if not using device_map
        if self.config.device != "cuda" or not quantization_config:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def get_module_by_name(self, name: str) -> nn.Module:
        """Get a module by its name path."""
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module
    
    def _register_hooks(self, layers: Optional[List[int]] = None):
        """Register forward hooks to capture activations."""
        self._remove_hooks()
        
        if layers is None:
            layers = list(range(len(self.model.transformer.h)))
            
        for layer_idx in layers:
            layer = self.model.transformer.h[layer_idx]
            
            def make_hook(idx):
                def hook_fn(module, input, output):
                    self.activation_cache[f"layer_{idx}"] = output[0].detach()
                return hook_fn
                
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)
            
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache.clear()
        
    def get_activations(self, inputs: torch.Tensor, layer: int) -> torch.Tensor:
        """Get activations at a specific layer."""
        self._register_hooks([layer])
        
        with torch.no_grad():
            _ = self.model(inputs)
            
        activations = self.activation_cache.get(f"layer_{layer}")
        self._remove_hooks()
        
        return activations
    
    def get_weights(self, layer: int, component: str) -> torch.Tensor:
        """Get weights for a specific component."""
        layer_module = self.model.transformer.h[layer]
        
        if component == "attention":
            # Get attention weights (Q, K, V projections)
            weights = {
                "q": layer_module.attn.q_proj.weight,
                "k": layer_module.attn.k_proj.weight,
                "v": layer_module.attn.v_proj.weight,
                "o": layer_module.attn.o_proj.weight,
            }
        elif component == "mlp":
            # Get MLP weights
            weights = {
                "up": layer_module.mlp.up_proj.weight,
                "gate": layer_module.mlp.gate_proj.weight,
                "down": layer_module.mlp.down_proj.weight,
            }
        else:
            raise ValueError(f"Unknown component: {component}")
            
        return weights
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> Any:
        """Forward pass through the model."""
        return self.model(inputs, **kwargs)
    
    def patch_activations(self, inputs: torch.Tensor, layer: int,
                         new_activations: torch.Tensor) -> Any:
        """Patch activations at a specific layer (CMAP)."""
        # Create a forward hook that replaces activations
        def patch_hook(module, input, output):
            return (new_activations,) + output[1:]
            
        layer_module = self.model.transformer.h[layer]
        handle = layer_module.register_forward_hook(patch_hook)
        
        try:
            with torch.no_grad():
                output = self.model(inputs)
        finally:
            handle.remove()
            
        return output
    
    def get_lora_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Extract LoRA adapter weights if present."""
        if not self.config.is_lora:
            return None
            
        lora_weights = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_weights[name] = {
                    "A": module.lora_A.default.weight,
                    "B": module.lora_B.default.weight,
                    "scaling": module.scaling.get("default", 1.0)
                }
                
        return lora_weights
    
    def disable_lora(self):
        """Temporarily disable LoRA adapters for base model inference."""
        if self.config.is_lora:
            self.model.disable_adapter_layers()
            
    def enable_lora(self):
        """Re-enable LoRA adapters."""
        if self.config.is_lora:
            self.model.enable_adapter_layers()


class DualModelManager:
    """Manage base and fine-tuned models for efficient differential analysis."""
    
    def __init__(self, base_config: ModelConfig, ft_config: ModelConfig):
        self.base_config = base_config
        self.ft_config = ft_config
        
        # Check if we can use LoRA optimization
        self.use_lora_optimization = (
            ft_config.is_lora and 
            base_config.model_name == ft_config.model_name
        )
        
        self.base_model = None
        self.ft_model = None
        
    def load_models(self):
        """Load both models with memory optimization."""
        if self.use_lora_optimization:
            logger.info("Using LoRA optimization - loading single base model")
            # Load base model once
            self.base_model = HuggingFaceModelWrapper(self.base_config)
            self.base_model.load_model()
            
            # FT model shares base weights
            self.ft_model = self.base_model
            
        else:
            logger.info("Loading separate base and fine-tuned models")
            # Load both models separately
            self.base_model = HuggingFaceModelWrapper(self.base_config)
            self.base_model.load_model()
            
            self.ft_model = HuggingFaceModelWrapper(self.ft_config)
            self.ft_model.load_model()
            
    def get_base_activations(self, inputs: torch.Tensor, layer: int) -> torch.Tensor:
        """Get base model activations."""
        if self.use_lora_optimization:
            self.base_model.disable_lora()
            activations = self.base_model.get_activations(inputs, layer)
            self.base_model.enable_lora()
            return activations
        else:
            return self.base_model.get_activations(inputs, layer)
            
    def get_ft_activations(self, inputs: torch.Tensor, layer: int) -> torch.Tensor:
        """Get fine-tuned model activations."""
        return self.ft_model.get_activations(inputs, layer)
    
    def get_activation_diff(self, inputs: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute activation difference at a layer."""
        base_acts = self.get_base_activations(inputs, layer)
        ft_acts = self.get_ft_activations(inputs, layer)
        return ft_acts - base_acts
    
    def get_weight_diff(self, layer: int, component: str) -> Dict[str, torch.Tensor]:
        """Compute weight differences for a component."""
        base_weights = self.base_model.get_weights(layer, component)
        
        if self.use_lora_optimization:
            # For LoRA, the difference is just the adapter weights
            lora_weights = self.ft_model.get_lora_weights()
            diff_weights = {}
            
            for key in base_weights:
                # Find corresponding LoRA weights
                module_name = f"transformer.h.{layer}.{component}.{key}_proj"
                if module_name in lora_weights:
                    lora = lora_weights[module_name]
                    # Compute effective weight difference: alpha * A @ B
                    diff = lora["scaling"] * (lora["A"] @ lora["B"])
                    diff_weights[key] = diff
                else:
                    diff_weights[key] = torch.zeros_like(base_weights[key])
                    
            return diff_weights
        else:
            ft_weights = self.ft_model.get_weights(layer, component)
            return {k: ft_weights[k] - base_weights[k] for k in base_weights}


class ModelRegistry:
    """Registry for managing multiple model instances."""
    
    def __init__(self):
        self.models = {}
        self.managers = {}
        
    def register_model(self, name: str, config: ModelConfig):
        """Register a model configuration."""
        if name not in self.models:
            wrapper = HuggingFaceModelWrapper(config)
            wrapper.load_model()
            self.models[name] = wrapper
            
    def register_pair(self, pair_name: str, base_name: str, ft_name: str):
        """Register a base-finetuned pair for differential analysis."""
        if base_name not in self.models or ft_name not in self.models:
            raise ValueError("Models must be registered first")
            
        manager = DualModelManager(
            self.models[base_name].config,
            self.models[ft_name].config
        )
        manager.base_model = self.models[base_name]
        manager.ft_model = self.models[ft_name]
        
        self.managers[pair_name] = manager
        
    def get_model(self, name: str) -> HuggingFaceModelWrapper:
        """Get a registered model."""
        return self.models.get(name)
    
    def get_manager(self, pair_name: str) -> DualModelManager:
        """Get a model pair manager."""
        return self.managers.get(pair_name)