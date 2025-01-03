from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,    
    StableDiffusionXLPipeline,
    FluxPipeline,
    CogVideoXPipeline,
    MochiPipeline,
    AnimateDiffPipeline,
    EulerDiscreteScheduler
)

from .model_utils import load_annimatediff_motion_adapter


TARGET_IMAGE_SIZE: tuple[int, int] = (256, 256)

MAINNET_UID = 34
TESTNET_UID = 168

# Project constants
MAINNET_WANDB_PROJECT: str = 'bitmind-subnet'
TESTNET_WANDB_PROJECT: str = 'bitmind'
WANDB_ENTITY: str = 'bitmindai'

# Cache directories
HUGGINGFACE_CACHE_DIR: Path = Path.home() / '.cache' / 'huggingface'
SN34_CACHE_DIR: Path = Path.home() / '.cache' / 'sn34'
REAL_CACHE_DIR: Path = SN34_CACHE_DIR / 'real'
SYNTH_CACHE_DIR: Path = SN34_CACHE_DIR / 'synthetic'
REAL_VIDEO_CACHE_DIR: Path = REAL_CACHE_DIR / 'video'
REAL_IMAGE_CACHE_DIR: Path = REAL_CACHE_DIR / 'image'
SYNTH_VIDEO_CACHE_DIR: Path = SYNTH_CACHE_DIR / 'video'
SYNTH_IMAGE_CACHE_DIR: Path = SYNTH_CACHE_DIR / 'image'
VALIDATOR_INFO_PATH: Path = SN34_CACHE_DIR / 'validator.yaml'
SN34_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Update intervals in hours
VIDEO_ZIP_CACHE_UPDATE_INTERVAL = 3
IMAGE_PARQUET_CACHE_UPDATE_INTERVAL = 2
VIDEO_CACHE_UPDATE_INTERVAL = 1
IMAGE_CACHE_UPDATE_INTERVAL = 1

MAX_COMPRESSED_GB = 100
MAX_EXTRACTED_GB = 10

CHALLENGE_TYPE = {
    0: 'real',
    1: 'synthetic'
}

# Image datasets configuration
IMAGE_DATASETS: Dict[str, List[Dict[str, str]]] = {
    "real": [
        {"path": "bitmind/bm-real"},
        {"path": "bitmind/open-image-v7-256"},
        {"path": "bitmind/celeb-a-hq"},
        {"path": "bitmind/ffhq-256"},
        {"path": "bitmind/MS-COCO-unique-256"},
        {"path": "bitmind/AFHQ"},
        {"path": "bitmind/lfw"},
        {"path": "bitmind/caltech-256"},
        {"path": "bitmind/caltech-101"},
        {"path": "bitmind/dtd"}
    ]
}

VIDEO_DATASETS = {
    "real": [
        {
            "path": "nkp37/OpenVid-1M",
            "filetype": "zip"
        },
        {
            "path": "shangxd/imagenet-vidvrd",
            "filetype": "zip"
    	}
    ]
}


# Prompt generation model configurations
IMAGE_ANNOTATION_MODEL: str = "Salesforce/blip2-opt-6.7b-coco"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"



# Text-to-image model configurations
T2I_MODELS: Dict[str, Dict[str, Any]] = { 

# IT SAYS THIS ONE IS IN THERE TOO https://huggingface.co/datasets/bitmind/bm-diffusion, 37.9 GB
    
    "stabilityai/stable-diffusion-xl-base-1.0": {
        # https://huggingface.co/datasets/bitmind/AFHQ___stable-diffusion-xl-base-1.0, 1.63 GB
        # https://huggingface.co/datasets/bitmind/bm-subnet-weekly-stable-diffusion-xl-base-1.0-256, 5 GB
        # https://huggingface.co/datasets/bitmind/bm-sdxl, 12.6 GB
        
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16"
        },
        "use_autocast": False
    },
    "SG161222/RealVisXL_V4.0": {
        # https://huggingface.co/datasets/bitmind/AFHQ___RealVisXL_V4.0, 1.77 GB
        # https://huggingface.co/datasets/bitmind/bm-realvisxl (13 GB)
        
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "variant": "fp16"
        }
    },
    "Corcelio/mobius": {
        # https://huggingface.co/datasets/bitmind/bm-mobius 14.3 GB
        # https://huggingface.co/datasets/bitmind/bm-subnet-weekly-mobius-256 5.35 GB
        
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16
        }
    },
    "black-forest-labs/FLUX.1-dev": {
        # https://huggingface.co/datasets/bitmind/bm-subnet-weekly-FLUX.1-dev-256 # 76 MB
        # https://huggingface.co/datasets/bitmind/ffhq-256___FLUX.1-dev # 6.65 GB
        # https://huggingface.co/datasets/bitmind/celeb-a-hq___FLUX.1-dev # 2.65 GB
        
        "pipeline_cls": FluxPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_inference_steps": {"min": 50, "max": 125},
            "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu"),
            "height": [512, 768],
            "width": [512, 768]
        },
        "enable_model_cpu_offload": False
    },
    "prompthero/openjourney-v4" : {
        # https://huggingface.co/datasets/bitmind/google-image-scraper___0-to-1000___openjourney-v4, 111 MB #### THIS IS NOT ACCURATE, JUST A BAD DATA SCRAPE
        
        "pipeline_cls": StableDiffusionPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        }
    },
    "cagliostrolab/animagine-xl-3.1": {
        # https://huggingface.co/datasets/bitmind/google-image-scraper___0-to-1000___animagine-xl-3.1, 103 MB ### THIS IS NOT ACCURATE, JUST A BAD DATA SCRAPE
        
        "pipeline_cls": StableDiffusionXLPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.float16,
        }
    }
}
T2I_MODEL_NAMES: List[str] = list(T2I_MODELS.keys())


# Text-to-video model configurations
T2V_MODELS: Dict[str, Dict[str, Any]] = {
    "genmo/mochi-1-preview": {
        "pipeline_cls": MochiPipeline,
        "from_pretrained_args": {
            "variant": "bf16", 
            "torch_dtype": torch.bfloat16
        },
        "generate_args": {
            "num_frames": 84
        },
        #"enable_model_cpu_offload": True,
        "vae_enable_tiling": True
    },
    'THUDM/CogVideoX-5b': {
        "pipeline_cls": CogVideoXPipeline,
        "from_pretrained_args": {
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_videos_per_prompt": 1,
            "num_inference_steps": {"min": 50, "max": 125},
            "num_frames": 48,
        },
        "enable_model_cpu_offload": True,
        #"enable_sequential_cpu_offload": True,
        "vae_enable_slicing": True,
        "vae_enable_tiling": True
    },
    'ByteDance/AnimateDiff-Lightning': {
        "pipeline_cls": AnimateDiffPipeline,
        "from_pretrained_args": {
            "base": "emilianJR/epiCRealism",
            "torch_dtype": torch.bfloat16,
            "motion_adapter": load_annimatediff_motion_adapter()
        },
        "generate_args": {
            "guidance_scale": 2,
            "num_inference_steps": {"min": 50, "max": 125},
        },
        "scheduler": {
            "cls": EulerDiscreteScheduler,
            "from_config_args": {
                "timestep_spacing": "trailing",
                "beta_schedule": "linear"
            }
        }
    }
}
T2V_MODEL_NAMES: List[str] = list(T2V_MODELS.keys())

# Combined model configurations
T2VIS_MODELS: Dict[str, Dict[str, Any]] = {**T2I_MODELS, **T2V_MODELS}
T2VIS_MODEL_NAMES: List[str] = list(T2VIS_MODELS.keys())


def get_modality(model_name):
    if model_name in T2V_MODEL_NAMES:
        return 'video'
    elif model_name in T2I_MODEL_NAMES:
        return 'image'


def select_random_t2vis_model(modality: Optional[str] = None) -> str:
    """
    Select a random text-to-image or text-to-video model based on the specified
    modality.

    Args:
        modality: The type of model to select ('image', 'video', or 'random').
            If None or 'random', randomly chooses between image and video.

    Returns:
        The name of the selected model.

    Raises:
        NotImplementedError: If the specified modality is not supported.
    """
    if modality is None or modality == 'random':
        modality = np.random.choice(['image', 'video'])

    if modality == 'image':
        return np.random.choice(T2I_MODEL_NAMES)
    elif modality == 'video':
        return np.random.choice(T2V_MODEL_NAMES)
    else:
        raise NotImplementedError(f"Unsupported modality: {modality}")
