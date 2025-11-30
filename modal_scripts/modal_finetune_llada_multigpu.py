"""
LLaDA-V Multi-GPU Modal Fine-tuning Script
==========================================

Multi-GPU fine-tuning for LLaDA-V with FSDP and DDP support.

Supports:
- 2, 4, 8 GPUs (A100-80GB, H100, L40s)
- FSDP (Fully Sharded Data Parallel) - Best for LoRA and memory-constrained scenarios
- DDP (Distributed Data Parallel) - Vanilla PyTorch distributed training
- DDP + DeepSpeed ZeRO - Advanced optimization for large models

Usage:
    # Configure GPU settings at top of file:
    GPU_COUNT = 4
    GPU_TYPE = "A100-80GB"
    OUTPUT_BASE_DIR = "/data/llada_single_gpu"

    # Train with FSDP (recommended for LoRA)
    modal run modal_finetune_llada_multigpu.py::train_llada_fsdp --use-lora

    # Train with vanilla DDP (simple multi-GPU) - uses defaults
    modal run modal_finetune_llada_multigpu.py::train_llada_ddp

    # Train with DDP + DeepSpeed (advanced optimization)
    modal run modal_finetune_llada_multigpu.py::train_llada_ddp_deepspeed \
        --use-lora \
        --deepspeed-stage 2
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path
from modal import App, Image, Volume, Secret

# ==============================================================================
# MULTI-GPU CONFIGURATION - EDIT THESE VALUES
# ==============================================================================

GPU_COUNT = 4  # Options: 2, 4, 8 (currently configured for 4-GPU DDP training)
GPU_TYPE = "A100-80GB"  # Options: "A100-80GB", "H100", "L40s"

# Dataset Configuration - Update to match your volume structure
OUTPUT_BASE_DIR = "/data/llada_single_gpu"  # Base directory in volume
DATASETS_DIR = f"{OUTPUT_BASE_DIR}/datasets"
MODELS_DIR = f"{OUTPUT_BASE_DIR}/models"
CHECKPOINTS_DIR = f"{OUTPUT_BASE_DIR}/checkpoints"

# Derived configuration for Modal
MODAL_GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

# ==============================================================================
# MODAL SETUP
# ==============================================================================

app = App("llada-v-finetune-multigpu")
volume = Volume.from_name("llada-v-finetune-vol", create_if_missing=True)

# CUDA configuration
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Time constants
HOURS = 60 * 60

# Build Modal image with all dependencies
image = (
    Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "pkg-config",
        "clang",
        "libopenmpi-dev",
        "openmpi-bin",
    )
    .uv_pip_install(
        # PyTorch (install first for CUDA support)
        "torch",
        "torchvision",
        "torchaudio",
    )
    .uv_pip_install(
        # Core training dependencies
        "deepspeed==0.14.4",
        "peft==0.9.0",
        "accelerate==0.29.1",
        "transformers",
        "bitsandbytes==0.45.3",
        "tokenizers~=0.15.2",
        # Data and utilities
        "datasets==2.16.1",
        "sentencepiece~=0.1.99",
        "open_clip_torch",
        "timm",
        "hf_transfer",
        "hf_xet",
        # Vision libraries
        "opencv-python",
        "av",
        "decord",
        # Model utilities
        "einops==0.6.1",
        "einops-exts==0.0.4",
        "safetensors",
        # ML libraries
        "scikit-learn>=1.3.2",
        "scipy",
        "numpy",
        # API and serving
        "fastapi",
        "uvicorn",
        "gradio_client==0.2.9",
        # Logging and tracking
        "wandb",
        "pynvml",
        # Distributed training
        "mpi4py",
        # Utilities
        "shortuuid",
        "httpx==0.24.0",
        "ftfy",
        "requests",
        "tyro",
        "urllib3<=2.0.0",
        "pydantic==1.10.8",
        "jinja2==3.1.5",
        # Evaluation metrics
        "nltk",
        "jiwer",
        "python-Levenshtein",
        # General utilities
        "tqdm",
        "packaging",
        "pyyaml",  # For FSDP config generation
        "webdataset",
    )
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "train"),
        remote_path="/root",
        copy=True,
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "data/",
            "checkpoints/",
            "*.png",
            "*.jpg",
        ],
    )
    .env(
        {
            "HF_HOME": "/data/.cache",
            "WANDB_PROJECT": "llada-latex-ocr-multigpu",
        }
    )
)

huggingface_secret = Secret.from_name("adithya-hf-wandb")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def patch_transformers_for_pytorch26():
    """
    Patch transformers trainer for PyTorch 2.6+ compatibility.

    PyTorch 2.6+ defaults to weights_only=True in torch.load(), which breaks
    RNG state loading during checkpoint resumption. This patch fixes it.
    """
    import site
    from pathlib import Path

    # Find transformers installation
    site_packages = site.getsitepackages()
    trainer_file = None

    for sp in site_packages:
        candidate = Path(sp) / "transformers" / "trainer.py"
        if candidate.exists():
            trainer_file = str(candidate)
            break

    if not trainer_file:
        print("Warning: Could not find transformers/trainer.py to patch")
        return False

    # Apply patch
    with open(trainer_file, "r") as f:
        content = f.read()

    # Check if already patched
    if "weights_only=False" in content:
        print("✓ transformers/trainer.py already patched")
        return True

    # Patch the RNG state loading line
    original = "checkpoint_rng_state = torch.load(rng_file)"
    patched = "checkpoint_rng_state = torch.load(rng_file, weights_only=False)"

    if original in content:
        content = content.replace(original, patched)
        with open(trainer_file, "w") as f:
            f.write(content)
        print("✓ Patched transformers/trainer.py for PyTorch 2.6+ compatibility")
        return True
    else:
        print("Warning: Could not find target line in transformers/trainer.py")
        return False


def create_fsdp_config(gpu_count: int) -> str:
    """
    Create FSDP configuration and return path to temporary file.

    Args:
        gpu_count: Number of GPUs to use

    Returns:
        Path to temporary YAML config file
    """
    import yaml

    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "num_processes": gpu_count,
        "machine_rank": 0,
        "main_process_ip": "localhost",
        "main_process_port": 29500,
        "main_training_function": "main",
        "fsdp_config": {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_forward_prefetch": False,
            "fsdp_offload_params": False,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_use_orig_params": True,
        },
        "mixed_precision": "bf16",
    }

    # Write to temporary file
    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    )
    yaml.dump(config, config_file, default_flow_style=False)
    config_file.flush()
    config_file.close()

    print(f"✓ Created FSDP config: {config_file.name}")
    print(f"  - GPU count: {gpu_count}")
    print("  - Sharding: FULL_SHARD")
    print("  - Mixed precision: bf16")

    return config_file.name


def create_deepspeed_config(stage: int = 2) -> str:
    """
    Create DeepSpeed configuration and return path to temporary file.

    Args:
        stage: DeepSpeed ZeRO stage (2 or 3)

    Returns:
        Path to temporary JSON config file
    """
    config = {
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {"device": "none", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }

    # Write to temporary file
    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    )
    json.dump(config, config_file, indent=2)
    config_file.flush()
    config_file.close()

    print(f"✓ Created DeepSpeed config: {config_file.name}")
    print(f"  - ZeRO stage: {stage}")
    print("  - Mixed precision: bf16")

    return config_file.name


# ==============================================================================
# FSDP TRAINING
# ==============================================================================


@app.function(
    image=image,
    gpu=MODAL_GPU_CONFIG,
    volumes={"/data": volume},
    timeout=24 * HOURS,
    secrets=[huggingface_secret, Secret.from_dotenv()],
)
def train_llada_fsdp(
    dataset_path: str = f"{DATASETS_DIR}/latex_ocr_dataset_train",
    model_path: str = f"{MODELS_DIR}/LLaDA-V",
    output_dir: str = f"{CHECKPOINTS_DIR}/llada-latex-ocr-fsdp",
    use_lora: bool = True,
    use_hf_dataset: bool = True,
    mm_tunable_parts: str = "mm_mlp_adapter,mm_language_model",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    save_steps: int = 250,
    logging_steps: int = 1,
):
    """
    Train LLaDA-V using FSDP (Fully Sharded Data Parallel).

    FSDP shards model parameters, gradients, and optimizer states across GPUs,
    enabling training of larger models with reduced per-GPU memory.

    Best for: Large models, memory-constrained scenarios, LoRA training

    Args:
        dataset_path: Path to training dataset JSON
        model_path: Path to base model
        output_dir: Output directory for checkpoints
        use_lora: Whether to use LoRA (parameter-efficient fine-tuning)
        mm_tunable_parts: Which multimodal components to train
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps

    Returns:
        Dict with training status and output directory
    """
    import torch

    print("=" * 80)
    print("LLADA-V MULTI-GPU TRAINING - FSDP")
    print("=" * 80)
    print(f"GPU Configuration: {GPU_COUNT}x {GPU_TYPE}")
    print("Strategy: FSDP (Fully Sharded Data Parallel)")
    print(f"Training Mode: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print("=" * 80)
    print()

    # Environment setup
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["WANDB_PROJECT"] = "llada-latex-ocr-fsdp"
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["PYTHONPATH"] = "/root"

    # MPI/PMIx fixes for distributed training
    os.environ["PMIX_MCA_gds"] = "hash"
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

    # Patch transformers for PyTorch 2.6+
    patch_transformers_for_pytorch26()

    # Register safe globals for torch serialization
    try:
        torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
        torch.serialization.add_safe_globals(["numpy.core.multiarray._reconstruct"])
    except AttributeError:
        pass

    # Detect GPU configuration
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs")

    # Auto-adjust batch size for FSDP
    # FSDP has lower per-GPU memory overhead due to sharding
    if use_lora:
        per_device_train_batch_size = 4 if gpu_count <= 4 else 2
    else:
        per_device_train_batch_size = 2

    # Adjust gradient accumulation to maintain effective batch size of 16
    target_effective_batch = 16
    gradient_accumulation_steps = max(
        1, target_effective_batch // (per_device_train_batch_size * gpu_count)
    )

    print()
    print("Batch Configuration:")
    print(f"  Per-device batch size: {per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {per_device_train_batch_size * gpu_count * gradient_accumulation_steps}"
    )
    print()

    # Create FSDP configuration
    fsdp_config_path = create_fsdp_config(gpu_count)

    # Build training command
    command = [
        "accelerate",
        "launch",
        "--config_file",
        fsdp_config_path,
        "/root/llava/train/train_mem.py",
        # Model and data configuration
        "--model_name_or_path",
        model_path,
        "--version",
        "llava_llada",
    ]

    # Add dataset-specific arguments based on format
    if use_hf_dataset:
        # HuggingFace Arrow format
        hf_dataset_path = (
            dataset_path
            if not dataset_path.endswith(".json")
            else os.path.dirname(dataset_path)
        )
        command.extend(["--data_path", hf_dataset_path, "--use_hf_dataset", "True"])
    else:
        # Legacy JSON format
        command.extend(
            [
                "--data_path",
                dataset_path,
                "--image_folder",
                os.path.dirname(dataset_path),
            ]
        )

    command.extend(
        [
            "--vision_tower",
            model_path,
            # Vision-language configuration
            "--mm_tunable_parts",
            mm_tunable_parts,
            "--mm_projector_type",
            "mlp2x_gelu",
            "--mm_vision_select_layer",
            "-2",
            "--mm_use_im_start_end",
            "False",
            "--mm_use_im_patch_token",
            "False",
            "--mm_patch_merge_type",
            "spatial_unpad",
            "--image_aspect_ratio",
            "pad",
            "--group_by_modality_length",
            "True",
            # Training hyperparameters
            "--output_dir",
            output_dir,
            "--num_train_epochs",
            str(num_train_epochs),
            "--per_device_train_batch_size",
            str(per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--learning_rate",
            str(learning_rate),
            "--warmup_ratio",
            str(warmup_ratio),
            "--weight_decay",
            str(weight_decay),
            "--lr_scheduler_type",
            "cosine",
            # Precision and optimization
            "--bf16",
            "True",
            "--tf32",
            "True",
            "--model_max_length",
            "2048",
            "--gradient_checkpointing",
            "True",
            "--attn_implementation",
            "sdpa",
            "--optim",
            "adamw_8bit",
            # Data loading
            "--dataloader_num_workers",
            "4",
            "--lazy_preprocess",
            "True",
            "--dataloader_drop_last",
            "True",
            # Checkpointing and logging
            "--save_strategy",
            "steps",
            "--save_steps",
            str(save_steps),
            "--save_total_limit",
            "25",
            "--logging_steps",
            str(logging_steps),
            "--report_to",
            "wandb",
            "--logging_nan_inf_filter",
            "False",
            "--logging_first_step",
            "True",
            "--evaluation_strategy",
            "no",
            "--use_conversation_mask",
            "False",
        ]
    )

    # Add LoRA flags if enabled
    if use_lora:
        command.extend(
            [
                "--lora_enable",
                "True",
                "--lora_r",
                str(lora_r),
                "--lora_alpha",
                str(lora_alpha),
                "--lora_dropout",
                str(lora_dropout),
            ]
        )

    print()
    print("Starting FSDP training...")
    print("Command:", " ".join(command[:10]), "...")
    print()

    # Run training
    subprocess.run(command, check=True, env=os.environ.copy())

    # Cleanup temp config
    try:
        os.unlink(fsdp_config_path)
    except Exception:
        pass

    # Commit volume
    volume.commit()

    print()
    print("=" * 80)
    print("✓ FSDP Training completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)

    return {
        "status": "completed",
        "output_dir": output_dir,
        "strategy": "fsdp",
        "gpu_count": gpu_count,
        "gpu_type": GPU_TYPE,
    }


# ==============================================================================
# DDP TRAINING (Vanilla - without DeepSpeed)
# ==============================================================================


@app.function(
    image=image,
    gpu=MODAL_GPU_CONFIG,
    volumes={"/data": volume},
    timeout=24 * HOURS,
    secrets=[huggingface_secret, Secret.from_dotenv()],
)
def train_llada_ddp(
    dataset_path: str = f"{DATASETS_DIR}/latex_ocr_dataset_train",
    model_path: str = f"{MODELS_DIR}/LLaDA-V",
    output_dir: str = f"{CHECKPOINTS_DIR}/llada-latex-ocr-ddp",
    use_lora: bool = False,  # Matching single-GPU config
    use_hf_dataset: bool = True,
    mm_tunable_parts: str = "mm_mlp_adapter,mm_language_model",
    num_train_epochs: int = 2,  # Matching single-GPU config
    per_device_train_batch_size: int = 8,  # Matching single-GPU config
    gradient_accumulation_steps: int = 8,  # Matching single-GPU config
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.03,  # Matching single-GPU config
    weight_decay: float = 0.0,  # Matching single-GPU config
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    save_steps: int = 250,
    logging_steps: int = 1,
):
    """
    Train LLaDA-V using vanilla DDP (Distributed Data Parallel).

    Pure DDP without DeepSpeed - replicates the full model on each GPU.
    Simpler setup, good for smaller models and debugging.

    Best for: Simple multi-GPU training, debugging, smaller models

    Args:
        dataset_path: Path to training dataset JSON
        model_path: Path to base model
        output_dir: Output directory for checkpoints
        use_lora: Whether to use LoRA (parameter-efficient fine-tuning)
        mm_tunable_parts: Which multimodal components to train
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps

    Returns:
        Dict with training status and output directory
    """
    import torch

    print("=" * 80)
    print("LLADA-V MULTI-GPU TRAINING - DDP (Vanilla)")
    print("=" * 80)
    print(f"GPU Configuration: {GPU_COUNT}x {GPU_TYPE}")
    print("Strategy: DDP (Distributed Data Parallel) - No DeepSpeed")
    print(f"Training Mode: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print("=" * 80)
    print()

    # Environment setup
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["WANDB_PROJECT"] = "llada-latex-ocr"
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["PYTHONPATH"] = "/root"

    # MPI/PMIx fixes for distributed training
    os.environ["PMIX_MCA_gds"] = "hash"
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

    # NCCL optimization for multi-GPU
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_DEBUG"] = "WARN"

    # Patch transformers for PyTorch 2.6+
    patch_transformers_for_pytorch26()

    # Register safe globals for torch serialization
    try:
        torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
        torch.serialization.add_safe_globals(["numpy.core.multiarray._reconstruct"])
    except AttributeError:
        pass

    # Detect GPU configuration
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs")

    # Use user-provided batch configuration (matching single-GPU settings)
    print()
    print("Batch Configuration:")
    print(f"  Per-device batch size: {per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Number of GPUs: {gpu_count}")
    print(
        f"  Total effective batch size: {per_device_train_batch_size * gpu_count * gradient_accumulation_steps}"
    )
    print()

    # Build training command (no DeepSpeed)
    command = [
        "torchrun",
        "--nproc_per_node",
        str(gpu_count),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "localhost",
        "--master_port",
        "29500",
        "/root/llava/train/train_mem.py",
        # Model and data configuration
        "--model_name_or_path",
        model_path,
        "--version",
        "llava_llada",
    ]

    # Add dataset-specific arguments based on format
    if use_hf_dataset:
        # HuggingFace Arrow format
        hf_dataset_path = (
            dataset_path
            if not dataset_path.endswith(".json")
            else os.path.dirname(dataset_path)
        )
        command.extend(["--data_path", hf_dataset_path, "--use_hf_dataset", "True"])
    else:
        # Legacy JSON format
        command.extend(
            [
                "--data_path",
                dataset_path,
                "--image_folder",
                os.path.dirname(dataset_path),
            ]
        )

    command.extend(
        [
            "--vision_tower",
            model_path,
            # Vision-language configuration
            "--mm_tunable_parts",
            mm_tunable_parts,
            "--mm_projector_type",
            "mlp2x_gelu",
            "--mm_vision_select_layer",
            "-2",
            "--mm_use_im_start_end",
            "False",
            "--mm_use_im_patch_token",
            "False",
            "--mm_patch_merge_type",
            "spatial_unpad",
            "--image_aspect_ratio",
            "pad",
            "--group_by_modality_length",
            "True",
            # Training hyperparameters
            "--output_dir",
            output_dir,
            "--num_train_epochs",
            str(num_train_epochs),
            "--per_device_train_batch_size",
            str(per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--learning_rate",
            str(learning_rate),
            "--warmup_ratio",
            str(warmup_ratio),
            "--weight_decay",
            str(weight_decay),
            "--lr_scheduler_type",
            "cosine",
            # Precision and optimization
            "--bf16",
            "True",
            "--tf32",
            "True",
            "--model_max_length",
            "2048",
            "--gradient_checkpointing",
            "True",
            "--attn_implementation",
            "sdpa",
            "--optim",
            "adamw_8bit",
            # Data loading
            "--dataloader_num_workers",
            "4",
            "--lazy_preprocess",
            "True",
            "--dataloader_drop_last",
            "True",
            # Checkpointing and logging
            "--save_strategy",
            "steps",
            "--save_steps",
            str(save_steps),
            "--save_total_limit",
            "25",
            "--logging_steps",
            str(logging_steps),
            "--report_to",
            "wandb",
            "--logging_nan_inf_filter",
            "False",
            "--logging_first_step",
            "True",
            "--evaluation_strategy",
            "no",
            "--use_conversation_mask",
            "False",
        ]
    )

    # Add LoRA flags if enabled
    if use_lora:
        command.extend(
            [
                "--lora_enable",
                "True",
                "--lora_r",
                str(lora_r),
                "--lora_alpha",
                str(lora_alpha),
                "--lora_dropout",
                str(lora_dropout),
            ]
        )

    print()
    print("Starting vanilla DDP training...")
    print("Command:", " ".join(command[:10]), "...")
    print()

    # Run training
    subprocess.run(command, check=True, env=os.environ.copy())

    # Commit volume
    volume.commit()

    print()
    print("=" * 80)
    print("✓ DDP Training completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)

    return {
        "status": "completed",
        "output_dir": output_dir,
        "strategy": "ddp",
        "gpu_count": gpu_count,
        "gpu_type": GPU_TYPE,
    }


# ==============================================================================
# DDP TRAINING (with DeepSpeed ZeRO)
# ==============================================================================


@app.function(
    image=image,
    gpu=MODAL_GPU_CONFIG,
    volumes={"/data": volume},
    timeout=24 * HOURS,
    secrets=[huggingface_secret, Secret.from_dotenv()],
)
def train_llada_ddp_deepspeed(
    dataset_path: str = f"{DATASETS_DIR}/latex_ocr_dataset_train",
    model_path: str = f"{MODELS_DIR}/LLaDA-V",
    output_dir: str = f"{CHECKPOINTS_DIR}/llada-latex-ocr-ddp-deepspeed",
    use_lora: bool = True,
    use_hf_dataset: bool = True,
    mm_tunable_parts: str = "mm_mlp_adapter,mm_language_model",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    save_steps: int = 250,
    logging_steps: int = 1,
    deepspeed_stage: int = 2,
):
    """
    Train LLaDA-V using DDP with DeepSpeed ZeRO.

    DDP replicates the full model on each GPU. DeepSpeed ZeRO adds:
    - Stage 2: Shards optimizer states and gradients
    - Stage 3: Shards parameters, gradients, and optimizer states

    Best for: Maximum throughput, full fine-tuning, smaller models

    Args:
        dataset_path: Path to training dataset JSON
        model_path: Path to base model
        output_dir: Output directory for checkpoints
        use_lora: Whether to use LoRA (parameter-efficient fine-tuning)
        mm_tunable_parts: Which multimodal components to train
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        deepspeed_stage: DeepSpeed ZeRO stage (2 or 3)

    Returns:
        Dict with training status and output directory
    """
    import torch

    print("=" * 80)
    print("LLADA-V MULTI-GPU TRAINING - DDP")
    print("=" * 80)
    print(f"GPU Configuration: {GPU_COUNT}x {GPU_TYPE}")
    print(f"Strategy: DDP with DeepSpeed ZeRO-{deepspeed_stage}")
    print(f"Training Mode: {'LoRA' if use_lora else 'Full Fine-tuning'}")
    print("=" * 80)
    print()

    # Environment setup
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["WANDB_PROJECT"] = "llada-latex-ocr-ddp"
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["PYTHONPATH"] = "/root"

    # MPI/PMIx fixes for distributed training
    os.environ["PMIX_MCA_gds"] = "hash"
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

    # NCCL optimization for multi-GPU
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_DEBUG"] = "WARN"

    # Patch transformers for PyTorch 2.6+
    patch_transformers_for_pytorch26()

    # Register safe globals for torch serialization
    try:
        torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
        torch.serialization.add_safe_globals(["numpy.core.multiarray._reconstruct"])
    except AttributeError:
        pass

    # Detect GPU configuration
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs")

    # Auto-adjust batch size for DDP
    # DDP replicates model, so memory usage is higher per GPU
    if use_lora:
        per_device_train_batch_size = 2
    else:
        per_device_train_batch_size = 1

    # Adjust gradient accumulation to maintain effective batch size of 16
    target_effective_batch = 16
    gradient_accumulation_steps = max(
        1, target_effective_batch // (per_device_train_batch_size * gpu_count)
    )

    print()
    print("Batch Configuration:")
    print(f"  Per-device batch size: {per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {per_device_train_batch_size * gpu_count * gradient_accumulation_steps}"
    )
    print()

    # Create DeepSpeed configuration
    deepspeed_config_path = create_deepspeed_config(deepspeed_stage)

    # Build training command
    command = [
        "torchrun",
        "--nproc_per_node",
        str(gpu_count),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "localhost",
        "--master_port",
        "29500",
        "/root/llava/train/train_mem.py",
        # DeepSpeed configuration
        "--deepspeed",
        deepspeed_config_path,
        # Model and data configuration
        "--model_name_or_path",
        model_path,
        "--version",
        "llava_llada",
    ]

    # Add dataset-specific arguments based on format
    if use_hf_dataset:
        # HuggingFace Arrow format
        hf_dataset_path = (
            dataset_path
            if not dataset_path.endswith(".json")
            else os.path.dirname(dataset_path)
        )
        command.extend(["--data_path", hf_dataset_path, "--use_hf_dataset", "True"])
    else:
        # Legacy JSON format
        command.extend(
            [
                "--data_path",
                dataset_path,
                "--image_folder",
                os.path.dirname(dataset_path),
            ]
        )

    command.extend(
        [
            "--vision_tower",
            model_path,
            # Vision-language configuration
            "--mm_tunable_parts",
            mm_tunable_parts,
            "--mm_projector_type",
            "mlp2x_gelu",
            "--mm_vision_select_layer",
            "-2",
            "--mm_use_im_start_end",
            "False",
            "--mm_use_im_patch_token",
            "False",
            "--mm_patch_merge_type",
            "spatial_unpad",
            "--image_aspect_ratio",
            "pad",
            "--group_by_modality_length",
            "True",
            # Training hyperparameters
            "--output_dir",
            output_dir,
            "--num_train_epochs",
            str(num_train_epochs),
            "--per_device_train_batch_size",
            str(per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--learning_rate",
            str(learning_rate),
            "--warmup_ratio",
            str(warmup_ratio),
            "--weight_decay",
            str(weight_decay),
            "--lr_scheduler_type",
            "cosine",
            # Precision and optimization
            "--bf16",
            "True",
            "--tf32",
            "True",
            "--model_max_length",
            "2048",
            "--gradient_checkpointing",
            "True",
            "--attn_implementation",
            "sdpa",
            # Data loading
            "--dataloader_num_workers",
            "4",
            "--lazy_preprocess",
            "True",
            "--dataloader_drop_last",
            "True",
            # Checkpointing and logging
            "--save_strategy",
            "steps",
            "--save_steps",
            str(save_steps),
            "--save_total_limit",
            "25",
            "--logging_steps",
            str(logging_steps),
            "--report_to",
            "wandb",
            "--logging_nan_inf_filter",
            "False",
            "--logging_first_step",
            "True",
            "--evaluation_strategy",
            "no",
            "--use_conversation_mask",
            "False",
        ]
    )

    # Add LoRA flags if enabled
    if use_lora:
        command.extend(
            [
                "--lora_enable",
                "True",
                "--lora_r",
                str(lora_r),
                "--lora_alpha",
                str(lora_alpha),
                "--lora_dropout",
                str(lora_dropout),
            ]
        )

    print()
    print("Starting DDP training with DeepSpeed...")
    print("Command:", " ".join(command[:10]), "...")
    print()

    # Run training
    subprocess.run(command, check=True, env=os.environ.copy())

    # Cleanup temp config
    try:
        os.unlink(deepspeed_config_path)
    except Exception:
        pass

    # Commit volume
    volume.commit()

    print()
    print("=" * 80)
    print("✓ DDP Training completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)

    return {
        "status": "completed",
        "output_dir": output_dir,
        "strategy": "ddp",
        "deepspeed_stage": deepspeed_stage,
        "gpu_count": gpu_count,
        "gpu_type": GPU_TYPE,
    }


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

"""
===============================================================================
LLADA-V MULTI-GPU TRAINING - 4 GPUs DDP Configuration
===============================================================================

VOLUME: llada-v-finetune-vol
DATASET PATH: /data/llada_single_gpu/datasets/latex_ocr_dataset_train (Arrow format)
GPU CONFIG: 4x A100-80GB with vanilla DDP

CONFIGURATION AT TOP OF FILE:
------------------------------
GPU_COUNT = 4
GPU_TYPE = "A100-80GB"
OUTPUT_BASE_DIR = "/data/llada_single_gpu"

STEP 1: Prepare Dataset (Arrow Format)
---------------------------------------
# Run from single-GPU script to prepare dataset in Arrow format
modal run modal_finetune_llada.py::download_and_prepare_dataset \\
    --save-format arrow

# This creates: /data/llada_single_gpu/datasets/latex_ocr_dataset_train/

STEP 2: Download Model
----------------------
modal run modal_finetune_llada.py::download_model

# This creates: /data/llada_single_gpu/models/LLaDA-V/

STEP 3: Run 4-GPU DDP Training
-------------------------------
# RECOMMENDED: Vanilla DDP with Arrow dataset (matching single-GPU config)
modal run modal_finetune_llada_multigpu.py::train_llada_ddp

# This will use the following defaults:
# - Dataset: /data/llada_single_gpu/datasets/latex_ocr_dataset_train (Arrow format)
# - GPUs: 4x A100-80GB
# - Per-device batch size: 8
# - Gradient accumulation: 8
# - Total effective batch size: 256 (8 * 4 * 8)
# - Epochs: 2
# - Learning rate: 2e-5
# - LoRA: Disabled (full fine-tuning)

# Output: /data/llada_single_gpu/checkpoints/llada-latex-ocr-ddp/


ALTERNATIVE TRAINING STRATEGIES:
---------------------------------

# FSDP Training (for larger models or memory constraints):
modal run modal_finetune_llada_multigpu.py::train_llada_fsdp \
    --use-lora

# DDP + DeepSpeed ZeRO-2 (advanced optimization):
modal run modal_finetune_llada_multigpu.py::train_llada_ddp_deepspeed \
    --deepspeed-stage 2


MONITORING:
-----------
# View logs in Modal dashboard
# WandB project: llada-latex-ocr
# Checkpoints saved every 500 steps


===============================================================================


MULTI-GPU TRAINING - OTHER CONFIGURATIONS:
-------------------------------------------

# Configure GPUs at top of file:
GPU_COUNT = 4  # Options: 2, 4, 8
GPU_TYPE = "A100-80GB"  # Options: "A100-80GB", "H100", "L40s"
OUTPUT_BASE_DIR = "/data/llada_single_gpu"  # Match your volume structure

# FSDP Training (Arrow format - RECOMMENDED):
modal run modal_finetune_llada_multigpu.py::train_llada_fsdp \
    --dataset-path /data/latex_ocr_dataset_train \
    --use-hf-dataset \
    --use-lora

# FSDP Training (JSON format):
modal run modal_finetune_llada_multigpu.py::train_llada_fsdp \
    --dataset-path /data/latex_ocr_dataset_train/dataset.json \
    --no-use-hf-dataset \
    --use-lora

# Vanilla DDP Training (Arrow format - RECOMMENDED):
modal run modal_finetune_llada_multigpu.py::train_llada_ddp \
    --dataset-path /data/llada_single_gpu/datasets/latex_ocr_dataset_train \
    --use-hf-dataset \
    --use-lora

# Vanilla DDP Training (JSON format):
modal run modal_finetune_llada_multigpu.py::train_llada_ddp \
    --dataset-path /data/latex_ocr_dataset_train/dataset.json \
    --no-use-hf-dataset \
    --use-lora

# DDP + DeepSpeed Training (Arrow format - RECOMMENDED):
modal run modal_finetune_llada_multigpu.py::train_llada_ddp_deepspeed \
    --dataset-path /data/latex_ocr_dataset_train \
    --use-hf-dataset \
    --use-lora \
    --deepspeed-stage 2

# DDP + DeepSpeed Training (JSON format):
modal run modal_finetune_llada_multigpu.py::train_llada_ddp_deepspeed \
    --dataset-path /data/latex_ocr_dataset_train/dataset.json \
    --no-use-hf-dataset \
    --use-lora \
    --deepspeed-stage 2


BENEFITS OF ARROW FORMAT:
--------------------------
- 2-5x faster loading (no disk I/O during training)
- Memory efficient (memory-mapped Arrow files)
- No corrupted image issues (filtered during preparation)
- Smaller storage footprint (compressed images)
- Backward compatible (JSON format still supported)
"""
