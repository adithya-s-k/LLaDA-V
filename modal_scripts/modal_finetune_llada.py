"""
LLaDA-V Modal Fine-tuning Script
=================================

Fine-tune LLaDA-V (diffusion-based vision-language model) on LaTeX OCR dataset.

Supports both LoRA and full fine-tuning modes with configurable components.

Usage:
    # 1. Download and prepare dataset
    modal run modal_finetune_llada.py::download_and_prepare_dataset --max-samples 10000

    modal run modal_finetune_llada.py::download_and_prepare_dataset --split validation --max-samples 100

    # 2. Download pre-trained model
    modal run modal_finetune_llada.py::download_model

    # 3. Train with LoRA
    modal run modal_finetune_llada.py::train_llada --use-lora True

    # 4. Train with full fine-tuning
    modal run modal_finetune_llada.py::train_llada --use-lora False

    # 5. Evaluate model
    modal run modal_finetune_llada.py::evaluate_model

    # 6. Test inference
    modal run modal_finetune_llada.py::inference_finetuned --image-path test.png
"""

import os
import subprocess
from pathlib import Path
from modal import App, Image, Volume, Secret

# ==============================================================================
# MODAL SETUP
# ==============================================================================

app = App("llada-v-finetune")
volume = Volume.from_name("llada-v-finetune-vol", create_if_missing=True)

# CUDA configuration (matching inference script)
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
        "cmake",  # Required for building sentencepiece
        "pkg-config",  # Required for building sentencepiece
        "clang",  # Required for building scikit-learn
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
        # Core training dependencies from pyproject.toml [train]
        "deepspeed==0.14.4",  # Optional: only used if --deepspeed-config is specified
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
        "scikit-learn>=1.3.2",  # Upgraded from 1.2.2 for Python 3.12 wheel support
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
        # Utilities from [standalone]
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
        "pyyaml",
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
            "WANDB_PROJECT": "llada-latex-ocr",
        }
    )
)

huggingface_secret = Secret.from_name("adithya-hf-wandb")

# ==============================================================================
# OUTPUT DIRECTORY CONFIGURATION - EDIT THIS TO ORGANIZE EXPERIMENTS
# ==============================================================================

OUTPUT_BASE_DIR = "/data/llada_single_gpu"  # Base directory for all outputs

# Derived paths
MODELS_DIR = f"{OUTPUT_BASE_DIR}/models"
DATASETS_DIR = f"{OUTPUT_BASE_DIR}/datasets"
CHECKPOINTS_DIR = f"{OUTPUT_BASE_DIR}/checkpoints"

# ==============================================================================
# DATASET PREPARATION
# ==============================================================================


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[huggingface_secret],
    timeout=4 * HOURS,
)
def download_and_prepare_dataset(
    dataset_name: str = "lukbl/LaTeX-OCR-dataset",
    split: str = "train",
    max_samples: int = None,
    save_format: str = "arrow",
):
    """
    Download and prepare LaTeX OCR dataset in LLaDA-V conversation format.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split ("train" or "validation")
        max_samples: Optional limit for testing (None for all samples)
        save_format: Save format ("arrow" for HuggingFace Arrow format, "json" for legacy JSON+images)

    Returns:
        Dict with status, dataset path, and sample count
    """
    from datasets import load_dataset
    from PIL import Image
    from io import BytesIO

    target_dataset_path = f"{DATASETS_DIR}/latex_ocr_dataset_{split}"

    print(f"Loading {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples")

    def is_valid_sample(example):
        """Check if sample is valid"""
        try:
            # Check text
            latex_text = example.get("text", "").strip()
            if not latex_text:
                return False

            # Check image exists
            image_data = example.get("image")
            if image_data is None:
                return False

            # Try to load image
            if isinstance(image_data, dict):
                if "bytes" in image_data:
                    image = Image.open(BytesIO(image_data["bytes"]))
                elif "path" in image_data:
                    image = Image.open(image_data["path"])
                else:
                    return False
            elif isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
            elif hasattr(image_data, "read"):
                image = Image.open(image_data)
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                return False

            # Validate dimensions
            if image.size[0] == 0 or image.size[1] == 0:
                return False

            return True
        except Exception:
            return False

    def transform_sample(example, idx):
        """Transform sample to LLaDA-V format"""
        latex_text = example["text"].strip()
        image_data = example["image"]

        # Load and convert image
        if isinstance(image_data, dict):
            if "bytes" in image_data:
                image = Image.open(BytesIO(image_data["bytes"]))
            else:
                image = Image.open(image_data["path"])
        elif isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data))
        elif hasattr(image_data, "read"):
            image = Image.open(image_data)
        else:
            image = image_data

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # LLaDA-V conversation format
        return {
            "id": f"latex_ocr_{split}_{idx}",
            "image": image,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nConvert this mathematical expression to LaTeX code.",
                },
                {
                    "from": "gpt",
                    "value": latex_text,
                },
            ],
        }

    # Step 1: Filter valid samples manually to avoid automatic image decoding
    print("Step 1/2: Filtering valid samples...")
    from tqdm import tqdm

    valid_indices = []
    print("  Checking samples for validity...")

    # Manually iterate to avoid automatic PIL decoding
    for idx in tqdm(range(len(dataset)), desc="Filtering"):
        try:
            sample = dataset[idx]
            if is_valid_sample(sample):
                valid_indices.append(idx)
        except Exception:
            # Skip corrupted samples
            continue

    # Select only valid samples
    valid_dataset = dataset.select(valid_indices)
    print(f"  Kept {len(valid_dataset)}/{len(dataset)} valid samples")

    # Step 2: Transform to LLaDA-V format
    print("Step 2/2: Transforming to LLaDA-V format...")
    filtered_dataset = valid_dataset.map(
        transform_sample,
        with_indices=True,
        remove_columns=valid_dataset.column_names,
        desc="Transforming",
    )

    if len(filtered_dataset) == 0:
        raise ValueError("No valid samples after filtering")

    import os

    success_rate = len(filtered_dataset) / len(dataset) * 100

    # Save dataset based on format
    if save_format == "arrow":
        # Save as HuggingFace Dataset (Arrow format with embedded images)
        os.makedirs(target_dataset_path, exist_ok=True)

        # Save to disk in Arrow format
        # HuggingFace datasets automatically infers the schema and encodes PIL Images
        print(f"Saving to {target_dataset_path} in Arrow format...")
        filtered_dataset.save_to_disk(target_dataset_path)

        volume.commit()

        print(
            f"✓ Saved {len(filtered_dataset)}/{len(dataset)} samples ({success_rate:.1f}%) in Arrow format"
        )
        print(f"  Dataset directory: {target_dataset_path}")

        return {
            "status": "completed",
            "dataset_path": target_dataset_path,
            "samples": len(filtered_dataset),
            "success_rate": f"{success_rate:.1f}%",
            "format": "arrow",
        }
    else:
        # Save dataset in JSON format for LLaDA-V training (legacy)
        import json

        os.makedirs(target_dataset_path, exist_ok=True)
        images_dir = os.path.join(target_dataset_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Convert dataset to JSON with saved images
        json_data = []
        for sample in tqdm(filtered_dataset, desc="Saving images"):
            # Save image to file
            image = sample["image"]
            image_filename = f"{sample['id']}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            image.save(image_path, "JPEG")

            # Create JSON entry with relative image path
            json_entry = {
                "id": sample["id"],
                "image": os.path.join("images", image_filename),
                "conversations": sample["conversations"],
            }
            json_data.append(json_entry)

        # Save JSON file
        json_path = os.path.join(target_dataset_path, "dataset.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        volume.commit()

        print(
            f"✓ Saved {len(filtered_dataset)}/{len(dataset)} samples ({success_rate:.1f}%) in JSON format"
        )
        print(f"  JSON file: {json_path}")
        print(f"  Images directory: {images_dir}")

        return {
            "status": "completed",
            "dataset_path": json_path,  # Return JSON file path, not directory
            "samples": len(filtered_dataset),
            "success_rate": f"{success_rate:.1f}%",
            "format": "json",
        }


# ==============================================================================
# MODEL DOWNLOAD
# ==============================================================================


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[huggingface_secret],
    timeout=2 * HOURS,
)
def download_model(force_redownload: bool = False):
    """
    Download pre-trained LLaDA-V model from HuggingFace.

    This model includes:
    - LLaDA-8B language model (diffusion-based)
    - SigLIP vision tower (already fine-tuned)
    - MM projector (vision-to-language alignment)

    Args:
        force_redownload: Force re-download even if model exists

    Returns:
        Dict with status and model path
    """
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    from huggingface_hub import snapshot_download
    import shutil

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = f"{MODELS_DIR}/LLaDA-V"
    config_path = os.path.join(model_path, "config.json")

    # Check if download is broken (lock files but no actual config)
    is_broken = os.path.exists(model_path) and not os.path.exists(config_path)

    if force_redownload or is_broken:
        if os.path.exists(model_path):
            print(f"Removing broken/old download at {model_path}...")
            shutil.rmtree(model_path)

    if not os.path.exists(model_path):
        print("Downloading LLaDA-V from HuggingFace (GSAI-ML/LLaDA-V)...")
        print("This includes:")
        print("  - LLaDA-8B language model")
        print("  - SigLIP vision tower")
        print("  - MM projector")
        print()

        snapshot_download(
            repo_id="GSAI-ML/LLaDA-V",
            local_dir=model_path,
            token=os.environ.get("HF_TOKEN"),
        )

    # Verify config exists
    if not os.path.exists(config_path):
        print(f"Error: config.json not found at {config_path}")
        raise FileNotFoundError("Model download failed - config.json missing")

    volume.commit()
    print(f"✓ Model ready at {model_path}")

    return {
        "status": "completed",
        "model_path": model_path,
    }


# ==============================================================================
# TRAINING
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


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    timeout=24 * HOURS,
    secrets=[huggingface_secret, Secret.from_dotenv()],
)
def train_llada(
    dataset_path=f"{DATASETS_DIR}/latex_ocr_dataset_train",
    model_path=f"{MODELS_DIR}/LLaDA-V",
    output_dir=f"{CHECKPOINTS_DIR}/llada-latex-ocr-full",
    # Training mode
    use_lora=False,  # True for LoRA, False for full fine-tuning
    # Tunable components (for LoRA: freeze vision tower, train adapter + LLM)
    mm_tunable_parts="mm_mlp_adapter,mm_language_model",
    # Training hyperparameters
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.0,
    # LoRA-specific parameters
    lora_r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    # DeepSpeed configuration (optional - only use for multi-GPU training)
    deepspeed_config=None,  # None = no DeepSpeed, "zero2" or "zero3" = enable DeepSpeed
    # Dataset format
    use_hf_dataset=True,  # True = HuggingFace Arrow format, False = legacy JSON format
    # Other settings
    save_steps=500,
    logging_steps=10,
    logging_nan_inf_filter=False,  # Log all values including NaN/Inf for debugging
):
    """
    Train LLaDA-V on LaTeX OCR dataset.

    Supports both LoRA and full fine-tuning modes with configurable components.

    Args:
        dataset_path: Path to prepared dataset (directory for Arrow format, JSON file for legacy)
        model_path: Path to pre-trained LLaDA-V model
        output_dir: Directory to save checkpoints
        use_lora: Enable LoRA fine-tuning (faster, smaller checkpoints)
        mm_tunable_parts: Components to train (comma-separated):
            - "mm_mlp_adapter" - Only projector (minimal training)
            - "mm_mlp_adapter,mm_language_model" - Adapter + LLM (recommended for LoRA)
            - "mm_vision_tower,mm_mlp_adapter,mm_language_model" - Full (only for non-LoRA)
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank (only used if use_lora=True)
        lora_alpha: LoRA alpha (only used if use_lora=True)
        lora_dropout: LoRA dropout (only used if use_lora=True)
        deepspeed_config: DeepSpeed config name (None=disabled, "zero2" or "zero3"=enabled)
        use_hf_dataset: Use HuggingFace Arrow format (True, recommended) or legacy JSON (False)

    Returns:
        Dict with status and output directory
    """
    # Set environment variables
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")
    os.environ["WANDB_PROJECT"] = "llada-latex-ocr"
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["PYTHONPATH"] = "/root"

    # Fix MPI/PMIx shared memory issues in containers
    os.environ["PMIX_MCA_gds"] = "hash"
    os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

    # Patch transformers for PyTorch 2.6+ checkpoint resumption compatibility
    patch_transformers_for_pytorch26()

    # Register numpy safe globals for torch serialization
    import torch

    try:
        torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
        torch.serialization.add_safe_globals(
            ["numpy.core.multiarray._reconstruct"]
        )  # Legacy numpy
    except AttributeError:
        # PyTorch < 2.6 doesn't have add_safe_globals
        pass

    # Optimize batch size for training mode to prevent OOM on A100-80GB
    # The model uses ~77-79GB even with LoRA, so we need smaller batches
    if per_device_train_batch_size > 2:
        original_batch_size = per_device_train_batch_size
        if use_lora:
            per_device_train_batch_size = 2  # LoRA: reduce to 2
        else:
            per_device_train_batch_size = 1  # Full fine-tuning: reduce to 1
        print(
            f"Note: Reducing batch size from {original_batch_size} to {per_device_train_batch_size} "
            f"for {'LoRA' if use_lora else 'full fine-tuning'} to prevent OOM on A100-80GB"
        )
        print(
            f"Effective batch size maintained at {per_device_train_batch_size * gradient_accumulation_steps} "
            f"via gradient accumulation"
        )

    print("=" * 80)
    print("LLADA-V FINE-TUNING ON LATEX OCR")
    print("=" * 80)
    print(f"Training mode: {'LoRA' if use_lora else 'Full fine-tuning'}")
    print(
        f"DeepSpeed: {'Enabled (' + deepspeed_config + ')' if deepspeed_config else 'Disabled (single-GPU)'}"
    )
    print(f"Tunable components: {mm_tunable_parts}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Batch size: {per_device_train_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(
        f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}"
    )
    print(f"Learning rate: {learning_rate}")
    if use_lora:
        print(f"LoRA rank: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print("=" * 80)
    print()

    # Verify dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run download_and_prepare_dataset first."
        )

    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run download_model first."
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if we'll be resuming from a checkpoint
    checkpoint_dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"📍 Found existing checkpoints - will resume from: {latest_checkpoint}")
    else:
        print("📍 No existing checkpoints - starting fresh training")

    # Build training command based on llada_v_finetune.sh
    command = [
        "python",
        "/root/llava/train/train_mem.py",
        # Model and data configuration
        "--model_name_or_path",
        model_path,
        "--version",
        "llava_llada",
        "--vision_tower",
        model_path,  # Vision tower is integrated in LLaDA-V
    ]

    # Add dataset-specific arguments based on format
    if use_hf_dataset:
        # HuggingFace Arrow format: expect directory path
        hf_dataset_path = dataset_path
        if dataset_path.endswith(".json"):
            # If user provided JSON path, use parent directory
            hf_dataset_path = os.path.dirname(dataset_path)

        command.extend(
            [
                "--data_path",
                hf_dataset_path,
                "--use_hf_dataset",
                "True",
            ]
        )
        # Note: --image_folder not needed for HF datasets (images are embedded)
    else:
        # Legacy JSON format: need both JSON file and image folder
        dataset_dir = os.path.dirname(dataset_path)
        command.extend(
            [
                "--data_path",
                dataset_path,
                "--image_folder",
                dataset_dir,
            ]
        )

    # Continue with common configuration
    command.extend(
        [
            # Vision-language configuration (from reference script)
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
            # Image processing (using simpler settings for LaTeX OCR)
            "--image_aspect_ratio",
            "pad",  # Simpler than anyres_max_4 for LaTeX equations
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
            "2048",  # Shorter than 8192 for LaTeX (equations are typically short)
            "--gradient_checkpointing",
            "True",
            "--attn_implementation",
            "sdpa",
            # Memory optimization: Use 8-bit AdamW to reduce optimizer memory usage
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
            "3",
            "--logging_steps",
            str(logging_steps),
            "--report_to",
            "wandb",
            # Logging configuration for better WandB metrics
            "--logging_nan_inf_filter",
            str(logging_nan_inf_filter),
            "--logging_first_step",
            "True",
            # Additional settings from reference
            "--evaluation_strategy",
            "no",
            "--use_conversation_mask",
            "False",
        ]
    )

    # Add DeepSpeed flag if enabled
    if deepspeed_config:
        deepspeed_config_path = f"/root/scripts/{deepspeed_config}.json"
        if not os.path.exists(deepspeed_config_path):
            raise FileNotFoundError(
                f"DeepSpeed config not found: {deepspeed_config_path}\n"
                f"Available configs: zero2.json, zero3.json, zero2_offload.json, zero3_offload.json"
            )
        command.extend(
            [
                "--deepspeed",
                deepspeed_config_path,
            ]
        )
        print(f"DeepSpeed enabled with config: {deepspeed_config_path}")
    else:
        print("DeepSpeed disabled - using standard PyTorch training")

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

    # Set environment
    env = os.environ.copy()

    print("Starting training...")
    print("Command: " + " ".join(command))
    print()

    # Run training
    result = subprocess.run(command, check=True, capture_output=False, env=env)

    # Commit volume
    volume.commit()

    print()
    print("=" * 80)
    print("✓ Training completed successfully!")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)

    return {
        "status": "completed",
        "output_dir": output_dir,
        "return_code": result.returncode,
        "training_mode": "lora" if use_lora else "full",
    }


# ==============================================================================
# MERGE LORA CHECKPOINT
# ==============================================================================


@app.function(
    image=image,
    gpu="L40s",
    volumes={"/data": volume},
    timeout=2 * HOURS,
    secrets=[huggingface_secret],
)
def merge_lora_checkpoint(
    checkpoint_path: str = f"{CHECKPOINTS_DIR}/llada-latex-ocr/checkpoint-500",
    base_model_path: str = f"{MODELS_DIR}/LLaDA-V",
    output_path: str = None,
    huggingface_repo_id: str = None,
    private: bool = True,
    merge_alpha: float = None,
):
    """
    Merge LoRA adapters with base model and optionally push to HuggingFace Hub.

    Args:
        checkpoint_path: Path to LoRA checkpoint directory
        base_model_path: Path to base LLaDA-V model
        output_path: Output directory for merged model (default: checkpoint-{N}-merged-alpha{value})
        huggingface_repo_id: HuggingFace repo ID to push to (e.g., "username/model-name")
        private: Whether to make HuggingFace repo private
        merge_alpha: Alpha scaling factor for merge (default: None = use training alpha)

    Returns:
        Dict with status, output_path, and optional hub_url
    """
    import sys

    sys.path.insert(0, "/root")

    import warnings
    import torch
    from llava.model.builder import load_pretrained_model
    from huggingface_hub import HfApi
    import shutil

    # Suppress PyTorch meta tensor warnings during model loading
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.nn.modules.module"
    )

    # Verify paths
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        raise ValueError(
            f"Not a LoRA checkpoint (missing adapter_config.json): {checkpoint_path}"
        )

    # Read LoRA configuration
    import json

    with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)

    lora_rank = adapter_config.get("r")
    training_alpha = adapter_config.get("lora_alpha")
    use_rslora = adapter_config.get("use_rslora", False)

    if lora_rank is None or training_alpha is None:
        raise ValueError(
            "Missing LoRA config in adapter_config.json (need 'r' and 'lora_alpha')"
        )

    # Determine merge alpha (default to training alpha)
    if merge_alpha is None:
        merge_alpha = training_alpha
        print(f"Using training alpha: {merge_alpha}")
    else:
        print(f"Using custom merge alpha: {merge_alpha}")

    # Validate merge_alpha
    if merge_alpha <= 0:
        raise ValueError(f"merge_alpha must be positive, got {merge_alpha}")

    # Calculate scaling factor
    if use_rslora:
        import math

        merge_scaling = merge_alpha / math.sqrt(lora_rank)
    else:
        merge_scaling = merge_alpha / lora_rank

    # Auto-generate output path if not specified (include alpha suffix)
    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path.rstrip("/"))
        parent_dir = os.path.dirname(checkpoint_path.rstrip("/"))
        output_path = os.path.join(
            parent_dir, f"{checkpoint_name}-merged-alpha{int(merge_alpha)}"
        )

    print("=" * 80)
    print("MERGE LORA CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base model: {base_model_path}")
    print(f"Output: {output_path}")
    if huggingface_repo_id:
        print(f"HuggingFace repo: {huggingface_repo_id} (private={private})")
    print("=" * 80)
    print()
    print(
        f"LoRA config: rank={lora_rank}, training_alpha={training_alpha}, merge_alpha={merge_alpha}"
    )
    print(f"Merge scaling factor: {merge_scaling:.6f}")
    print()

    # Load base model
    print("[1/4] Loading base model...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        base_model_path,
        None,
        "llava_llada",
        attn_implementation="sdpa",
        device_map="auto",
    )
    print("✓ Base model loaded")

    # Load LoRA adapter weights manually
    print("\n[2/4] Loading LoRA adapter weights...")
    from safetensors.torch import load_file

    # Find adapter weight files (.safetensors preferred, .bin fallback)
    adapter_files = [
        f
        for f in os.listdir(checkpoint_path)
        if f.startswith("adapter_model")
        and (f.endswith(".safetensors") or f.endswith(".bin"))
    ]

    if not adapter_files:
        raise FileNotFoundError(f"No adapter weight files found in {checkpoint_path}")

    # Load adapter weights
    adapter_weights = {}
    for adapter_file in adapter_files:
        file_path = os.path.join(checkpoint_path, adapter_file)
        if file_path.endswith(".safetensors"):
            adapter_weights.update(load_file(file_path))
        else:
            adapter_weights.update(torch.load(file_path, map_location="cpu"))

    print(f"✓ Loaded {len(adapter_weights)} adapter parameters")

    # Load non_lora_trainables.bin if present (mm_projector, etc.)
    non_lora_path = os.path.join(checkpoint_path, "non_lora_trainables.bin")
    non_lora_trainables = {}
    if os.path.exists(non_lora_path):
        print("Loading non-LoRA trainable weights...")
        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
        # Remove PEFT prefixes
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        print(f"✓ Loaded {len(non_lora_trainables)} non-LoRA parameters")

    # Manual merge with custom scaling
    print("\n[3/4] Merging LoRA weights with custom scaling...")
    base_state_dict = model.state_dict()

    # Group LoRA weights by layer
    lora_layers = {}
    for key in adapter_weights.keys():
        if "lora_A" in key or "lora_B" in key:
            # Extract base layer name
            if "lora_A" in key:
                base_key = key.split(".lora_A")[0]
                lora_type = "A"
            else:
                base_key = key.split(".lora_B")[0]
                lora_type = "B"

            if base_key not in lora_layers:
                lora_layers[base_key] = {}
            lora_layers[base_key][lora_type] = adapter_weights[key]

    print(f"Found {len(lora_layers)} layers with LoRA adapters")

    # Merge each layer
    merge_count = 0
    for base_key, lora_weights in lora_layers.items():
        if "A" not in lora_weights or "B" not in lora_weights:
            continue

        lora_A = lora_weights["A"]
        lora_B = lora_weights["B"]

        # Convert PEFT key to state dict key
        # Remove 'base_model.model.' prefix if present
        state_dict_key = base_key
        if state_dict_key.startswith("base_model.model."):
            state_dict_key = state_dict_key[17:]

        # Add .weight suffix
        state_dict_key = f"{state_dict_key}.weight"

        if state_dict_key in base_state_dict:
            base_weight = base_state_dict[state_dict_key]

            # Compute delta: (lora_B @ lora_A) * scaling
            device = base_weight.device
            dtype = base_weight.dtype

            # Use float32 for stability
            delta = (
                lora_B.to(device, dtype=torch.float32)
                @ lora_A.to(device, dtype=torch.float32)
            ) * merge_scaling

            # Add to base weight and convert back
            merged = base_weight.to(torch.float32) + delta
            base_state_dict[state_dict_key] = merged.to(dtype)
            merge_count += 1

    # Load non-LoRA trainables
    if non_lora_trainables:
        for key, value in non_lora_trainables.items():
            if key in base_state_dict:
                base_state_dict[key] = value

    # Load merged weights
    model.load_state_dict(base_state_dict, strict=False)
    merged_model = model

    print(f"✓ Merged {merge_count} LoRA layers with scaling factor {merge_scaling:.6f}")

    # Save merged model
    print(f"\n[4/5] Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Save model weights
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Copy generation config if exists
    gen_config_src = os.path.join(base_model_path, "generation_config.json")
    if os.path.exists(gen_config_src):
        shutil.copy(gen_config_src, output_path)

    # Copy custom config files (for llava_llada architecture)
    for config_file in ["configuration_llada.py"]:
        config_src = os.path.join(base_model_path, config_file)
        if os.path.exists(config_src):
            shutil.copy(config_src, output_path)

    # Save merge metadata
    merge_metadata = {
        "checkpoint_path": checkpoint_path,
        "base_model_path": base_model_path,
        "lora_config": {
            "r": lora_rank,
            "training_alpha": training_alpha,
            "merge_alpha": merge_alpha,
            "use_rslora": use_rslora,
            "merge_scaling": merge_scaling,
        },
        "merge_method": "manual",
    }

    with open(os.path.join(output_path, "merge_metadata.json"), "w") as f:
        json.dump(merge_metadata, f, indent=2)

    volume.commit()
    print("✓ Merged model saved")
    print(f"✓ Merge metadata saved to {output_path}/merge_metadata.json")

    # Push to HuggingFace Hub if requested
    hub_url = None
    if huggingface_repo_id:
        print(f"\n[5/5] Pushing to HuggingFace Hub: {huggingface_repo_id}...")
        os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

        try:
            # Create model card
            model_card = f"""---
tags:
- vision
- image-text-to-text
- llava
- latex
- ocr
library_name: transformers
pipeline_tag: image-text-to-text
---

# LLaDA-V Fine-tuned for LaTeX OCR

This model is a fine-tuned version of [GSAI-ML/LLaDA-V](https://huggingface.co/GSAI-ML/LLaDA-V) for LaTeX OCR tasks.

## Training Details

- **Base Model**: GSAI-ML/LLaDA-V
- **Fine-tuning Method**: LoRA
- **Dataset**: LaTeX OCR equations
- **Checkpoint**: {os.path.basename(checkpoint_path)}

## LoRA Configuration

- **Rank**: {lora_rank}
- **Training Alpha**: {training_alpha}
- **Merge Alpha**: {merge_alpha}
- **Scaling Factor**: {merge_scaling:.6f}
- **RS-LoRA**: {use_rslora}

## Usage

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from PIL import Image
import torch

# Load model
tokenizer, model, image_processor, _ = load_pretrained_model(
    "{huggingface_repo_id}",
    None,
    "llava_llada",
    device_map="auto",
)

# Load and process image
image = Image.open("equation.png")
text = "<image>\\nConvert this mathematical expression to LaTeX code."
input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
image_tensor = process_images([image], image_processor, model.config)

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids.unsqueeze(0).cuda(),
        images=image_tensor.cuda(),
        max_new_tokens=512,
    )
latex = tokenizer.decode(output[0], skip_special_tokens=True)
print(latex)
```

## Model Details

- **Architecture**: LLaDA-V (Diffusion-based Vision-Language Model)
- **Vision Encoder**: SigLIP-SO400M
- **Language Model**: LLaDA-8B
- **Training**: LoRA fine-tuning merged into base weights
"""

            # Save model card
            with open(os.path.join(output_path, "README.md"), "w") as f:
                f.write(model_card)

            # Push to hub
            merged_model.push_to_hub(
                huggingface_repo_id,
                private=private,
                token=os.environ["HF_TOKEN"],
            )
            tokenizer.push_to_hub(
                huggingface_repo_id,
                private=private,
                token=os.environ["HF_TOKEN"],
            )

            # Push additional files
            api = HfApi()
            for filename in ["generation_config.json", "configuration_llada.py"]:
                filepath = os.path.join(output_path, filename)
                if os.path.exists(filepath):
                    api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=huggingface_repo_id,
                        token=os.environ["HF_TOKEN"],
                    )

            hub_url = f"https://huggingface.co/{huggingface_repo_id}"
            print(f"✓ Pushed to HuggingFace Hub: {hub_url}")

        except Exception as e:
            print(f"✗ Failed to push to HuggingFace Hub: {e}")
            print("Model saved locally, but Hub upload failed")

    print("\n" + "=" * 80)
    print("✓ MERGE COMPLETED")
    print("=" * 80)
    print(f"Merged model: {output_path}")
    if hub_url:
        print(f"HuggingFace: {hub_url}")

    return {
        "status": "completed",
        "output_path": output_path,
        "hub_url": hub_url,
    }


# ==============================================================================
# PROCESS FULL CHECKPOINT
# ==============================================================================


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=1 * HOURS,
    secrets=[huggingface_secret],
)
def process_full_checkpoint(
    checkpoint_path: str = f"{CHECKPOINTS_DIR}/llada-latex-ocr-full/checkpoint-7500",
    output_path: str = None,
    huggingface_repo_id: str = None,
    private: bool = True,
):
    """
    Process full fine-tuned checkpoint for deployment.

    Creates a clean version with only inference-essential files by removing
    training artifacts (optimizer.pt, scheduler.pt, etc.).

    Args:
        checkpoint_path: Path to full fine-tuned checkpoint directory
        output_path: Output directory (default: checkpoint-{N}-processed)
        huggingface_repo_id: HuggingFace repo ID to push to
        private: Whether to make HuggingFace repo private

    Returns:
        Dict with status, output_path, and optional hub_url
    """
    import glob
    import json
    import shutil
    from datetime import datetime
    from huggingface_hub import HfApi

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Verify it's a full checkpoint (has model files, not adapter_config.json)
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        raise ValueError(
            "This is a LoRA checkpoint, use merge_lora_checkpoint() instead"
        )

    # Verify model files exist
    model_index = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(model_index):
        raise ValueError(
            f"Not a valid checkpoint (missing model.safetensors.index.json): {checkpoint_path}"
        )

    # Auto-generate output path if not specified
    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path.rstrip("/"))
        parent_dir = os.path.dirname(checkpoint_path.rstrip("/"))
        output_path = os.path.join(parent_dir, f"{checkpoint_name}-processed")

    print("=" * 80)
    print("PROCESS FULL FINE-TUNED CHECKPOINT")
    print("=" * 80)
    print(f"Input: {checkpoint_path}")
    print(f"Output: {output_path}")
    if huggingface_repo_id:
        print(f"HuggingFace repo: {huggingface_repo_id} (private={private})")
    print("=" * 80)
    print()

    # Files to keep (inference essential)
    ESSENTIAL_PATTERNS = [
        "model-*.safetensors",  # Model weights (sharded)
        "model.safetensors",  # Model weights (single file)
        "model.safetensors.index.json",  # Sharded model index
        "config.json",  # Model config
        "configuration_llada.py",  # Custom LLaDA config
        "tokenizer.json",  # Tokenizer vocabulary
        "tokenizer_config.json",  # Tokenizer config
        "special_tokens_map.json",  # Special tokens
        "generation_config.json",  # Generation params
    ]

    # Files to explicitly skip (training only)
    TRAINING_FILES = [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    ]

    print("[1/3] Copying essential files...")
    os.makedirs(output_path, exist_ok=True)

    copied_files = []
    total_size = 0

    # Copy files matching essential patterns
    for pattern in ESSENTIAL_PATTERNS:
        pattern_path = os.path.join(checkpoint_path, pattern)
        matched_files = glob.glob(pattern_path)

        for src_file in matched_files:
            filename = os.path.basename(src_file)
            dst_file = os.path.join(output_path, filename)

            shutil.copy2(src_file, dst_file)
            file_size = os.path.getsize(src_file)
            total_size += file_size
            copied_files.append(filename)

    print(
        f"✓ Copied {len(copied_files)} essential files ({total_size / 1024**3:.1f} GB)"
    )

    # Fix config.json to use HuggingFace vision tower instead of local path
    print("\n[2/4] Fixing vision tower reference in config...")
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Update vision tower to use HuggingFace model ID
        if config.get("mm_vision_tower") == f"{MODELS_DIR}/LLaDA-V":
            config["mm_vision_tower"] = "google/siglip2-so400m-patch14-384"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("✓ Updated mm_vision_tower to: google/siglip2-so400m-patch14-384")
        else:
            print(f"✓ Vision tower already correct: {config.get('mm_vision_tower')}")
    else:
        print("⚠ Warning: config.json not found")

    # Create processing metadata
    print("\n[3/4] Creating processing metadata...")

    # Calculate space saved
    checkpoint_size = sum(
        os.path.getsize(os.path.join(checkpoint_path, f))
        for f in os.listdir(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_path, f))
    )
    space_saved = checkpoint_size - total_size

    processing_metadata = {
        "source_checkpoint": checkpoint_path,
        "processed_date": datetime.now().isoformat(),
        "files_copied": len(copied_files),
        "total_size_gb": total_size / 1024**3,
        "space_saved_gb": space_saved / 1024**3,
        "files_removed": TRAINING_FILES,
        "essential_patterns": ESSENTIAL_PATTERNS,
    }

    with open(os.path.join(output_path, "processing_metadata.json"), "w") as f:
        json.dump(processing_metadata, f, indent=2)

    print(f"✓ Saved {space_saved / 1024**3:.1f} GB by removing training files")

    # Finalize
    print("\n[4/4] Finalizing...")
    volume.commit()

    hub_url = None
    if huggingface_repo_id:
        print(f"\nPushing to HuggingFace Hub: {huggingface_repo_id}...")

        os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=huggingface_repo_id,
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Note: {e}")

        # Create model card
        model_card = f"""---
tags:
- vision
- image-text-to-text
- llava
- latex
- ocr
library_name: transformers
pipeline_tag: image-text-to-text
---

# LLaDA-V Fine-tuned for LaTeX OCR

This is a processed deployment version of a full fine-tuned LLaDA-V model.

## Training Details

- **Base Model**: GSAI-ML/LLaDA-V
- **Fine-tuning Method**: Full Fine-tuning
- **Dataset**: LaTeX OCR equations
- **Source Checkpoint**: {os.path.basename(checkpoint_path)}

## Processing

This checkpoint has been processed for deployment:
- Training files removed (optimizer, scheduler, RNG state)
- Space saved: {space_saved / 1024**3:.1f} GB
- Only inference-essential files retained

## Usage

```python
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, _ = load_pretrained_model(
    "{huggingface_repo_id}",
    None,
    "llava_llada",
    device_map="auto",
)
```
"""

        with open(os.path.join(output_path, "README.md"), "w") as f:
            f.write(model_card)

        # Upload to Hub
        api.upload_folder(
            folder_path=output_path,
            repo_id=huggingface_repo_id,
            repo_type="model",
        )

        hub_url = f"https://huggingface.co/{huggingface_repo_id}"
        print(f"✓ Pushed to {hub_url}")

    print("\n" + "=" * 80)
    print("✓ Processing completed successfully!")
    print(f"Clean checkpoint: {output_path}")
    if hub_url:
        print(f"HuggingFace Hub: {hub_url}")
    print("=" * 80)

    result = {
        "status": "success",
        "output_path": output_path,
        "files_copied": len(copied_files),
        "size_gb": total_size / 1024**3,
        "space_saved_gb": space_saved / 1024**3,
    }

    if hub_url:
        result["hub_url"] = hub_url

    return result


# ==============================================================================
# EVALUATION
# ==============================================================================


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    timeout=4 * HOURS,
    secrets=[huggingface_secret, Secret.from_dotenv()],
)
def evaluate_model(
    model_path=f"{CHECKPOINTS_DIR}/llada-latex-ocr",
    dataset_path=f"{DATASETS_DIR}/latex_ocr_dataset_validation/dataset.json",
    output_dir="/data/evaluation_results",
    max_samples: int = 5,
    max_new_tokens: int = 512,
    batch_size: int = 1,  # Number of samples to process at once
):
    """
    Evaluate fine-tuned LLaDA-V model on validation set.

    RECOMMENDED: Merge LoRA checkpoints before evaluation using merge_lora_checkpoint().
    This function works best with merged models or HuggingFace repos.

    Calculates metrics:
    - Exact Match Accuracy
    - BLEU Score
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Edit Distance (Levenshtein)

    Args:
        model_path: Path to merged model or HuggingFace repo ID
                   (For LoRA checkpoints, merge first using merge_lora_checkpoint)
        dataset_path: Path to validation dataset
        output_dir: Directory to save evaluation results
        max_samples: Maximum samples to evaluate (None for all)
        max_new_tokens: Maximum tokens to generate
        batch_size: Number of samples to process at once (default=1 for stability)

    Returns:
        Dict with metrics and results file path
    """
    import sys

    sys.path.insert(0, "/root")  # Critical for llava imports

    import torch
    import numpy as np
    import warnings
    from tqdm import tqdm
    from datetime import datetime

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    from jiwer import wer, cer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import Levenshtein

    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LLADA-V MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print()

    # Load model with smart detection
    print("[1/3] Loading model...")

    # Auto-detect model type and load accordingly
    model_type = None
    base_model_path = f"{MODELS_DIR}/LLaDA-V"

    if os.path.exists(model_path):
        # Local path - check if it's a directory with checkpoints
        if os.path.isdir(model_path):
            checkpoint_dirs = [
                d
                for d in os.listdir(model_path)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(model_path, d))
            ]
            if checkpoint_dirs:
                # Has checkpoint subdirectories - use latest
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = checkpoint_dirs[-1]
                model_path = os.path.join(model_path, latest_checkpoint)
                print(f"Auto-selected latest checkpoint: {latest_checkpoint}")

        # Check if it's a LoRA checkpoint
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            model_type = "lora"
            print("Detected LoRA checkpoint")
            print(f"  Loading base model: {base_model_path}")
            print(f"  Loading LoRA adapters: {model_path}")
        else:
            model_type = "merged"
            print("Detected merged/full model")
    else:
        # Not a local path - assume HuggingFace Hub repo
        model_type = "huggingface"
        print(f"Loading from HuggingFace Hub: {model_path}")

    # Load model based on type
    if model_type == "lora":
        # Load base model + LoRA adapters
        # Note: model_name must include "lora" to trigger LoRA loading path
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path,  # LoRA checkpoint path
            base_model_path,  # Base model path
            "lora-llava-llada",  # Must include "lora" in name
            attn_implementation="sdpa",
            device_map=device,
        )
    else:
        # Load merged model or HuggingFace model
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path,
            None,
            "llava_llada",
            attn_implementation="sdpa",
            device_map=device,
        )

    model.eval()

    print("✓ Model loaded")
    print(f"  Type: {model_type}")
    print(f"  Device: {device}")
    print(f"  Max length: {max_length}")

    # Load dataset from JSON
    print("\n[2/3] Loading dataset...")
    import json
    from pathlib import Path as PathLib

    dataset_dir = str(PathLib(dataset_path).parent)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"✓ Evaluating {len(dataset)} samples with batch_size={batch_size}")

    # Evaluation loop with batching
    print("\n[3/3] Running evaluation...")
    results = []
    exact_matches = 0
    bleu_scores, edit_distances, cer_scores, wer_scores = [], [], [], []

    from PIL import Image as PILImage

    # Process in batches
    for batch_start in tqdm(
        range(0, len(dataset), batch_size), desc="Evaluating batches"
    ):
        batch_end = min(batch_start + batch_size, len(dataset))

        try:
            # Prepare batch
            batch_images = []
            batch_ground_truths = []
            batch_ids = []
            batch_image_sizes = []

            # Get individual samples from the batch
            for idx in range(batch_start, batch_end):
                sample = dataset[idx]
                image_path_rel = sample["image"]
                ground_truth = sample["conversations"][1]["value"]

                # Load image from file path (relative to dataset directory)
                image_path_abs = os.path.join(dataset_dir, image_path_rel)
                image = PILImage.open(image_path_abs).convert("RGB")

                batch_images.append(image)
                batch_ground_truths.append(ground_truth)
                batch_ids.append(sample["id"])
                batch_image_sizes.append(image.size)

            # Prepare inputs
            text = "<image>\nConvert this mathematical expression to LaTeX code."

            # Tokenize for each sample (they should all be the same length)
            batch_input_ids = []
            for _ in batch_images:
                input_ids = tokenizer_image_token(
                    text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                batch_input_ids.append(input_ids)

            # Stack into batch
            batch_input_ids = torch.stack(batch_input_ids).to(device)

            # Process images
            image_tensors = process_images(batch_images, image_processor, model.config)
            if image_tensors is None or len(image_tensors) == 0:
                print(
                    f"  Warning: process_images returned None for batch starting at {batch_start}"
                )
                continue

            image_tensors = [
                _image.to(dtype=torch.float16, device=device)
                for _image in image_tensors
            ]

            # Generate for batch
            with torch.no_grad():
                output_ids = model.generate(
                    batch_input_ids,
                    images=image_tensors,
                    image_sizes=batch_image_sizes,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )

            # Debug: Check output shapes
            if batch_start == 0:
                print(f"  Debug: batch_input_ids.shape = {batch_input_ids.shape}")
                print(f"  Debug: output_ids.shape = {output_ids.shape}")
                print(f"  Debug: num samples in batch = {len(batch_ground_truths)}")

            # Decode each output in the batch
            for output, ground_truth, sample_id in zip(
                output_ids, batch_ground_truths, batch_ids
            ):
                try:
                    prediction = tokenizer.decode(
                        output[batch_input_ids.shape[1] :], skip_special_tokens=True
                    ).strip()

                    # Calculate metrics
                    is_exact = prediction == ground_truth
                    if is_exact:
                        exact_matches += 1

                    bleu = sentence_bleu(
                        [list(ground_truth)],
                        list(prediction),
                        smoothing_function=SmoothingFunction().method1,
                    )
                    ed = Levenshtein.distance(prediction, ground_truth)
                    cer_v = cer(ground_truth, prediction)
                    wer_v = wer(ground_truth, prediction)

                    bleu_scores.append(bleu)
                    edit_distances.append(ed)
                    cer_scores.append(cer_v)
                    wer_scores.append(wer_v)

                    results.append(
                        {
                            "id": sample_id,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "exact_match": is_exact,
                            "bleu": bleu,
                            "edit_distance": ed,
                            "cer": cer_v,
                            "wer": wer_v,
                        }
                    )
                except Exception as e:
                    print(f"  Error processing sample {sample_id}: {e}")
                    continue

            # Print progress
            if len(results) % 20 == 0 and len(results) > 0:
                current_acc = exact_matches / len(results) * 100
                print(
                    f"  Progress: {len(results)}/{len(dataset)} | Exact Match: {current_acc:.2f}%"
                )

        except Exception as e:
            print(f"Error on batch starting at {batch_start}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Calculate final metrics
    total = len(results)

    if total == 0:
        print("\n❌ ERROR: All samples failed during evaluation!")
        print("No valid predictions were generated.")
        return {
            "status": "failed",
            "error": "All samples failed - no results to evaluate",
        }

    metrics = {
        "model": model_path,
        "model_type": model_type,
        "total_samples": total,
        "exact_match_accuracy": exact_matches / total * 100,
        "bleu_score_mean": float(np.mean(bleu_scores)),
        "bleu_score_std": float(np.std(bleu_scores)),
        "cer_mean": float(np.mean(cer_scores)),
        "cer_std": float(np.std(cer_scores)),
        "wer_mean": float(np.mean(wer_scores)),
        "wer_std": float(np.std(wer_scores)),
        "edit_distance_mean": float(np.mean(edit_distances)),
        "edit_distance_std": float(np.std(edit_distances)),
    }

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Samples: {total}")
    print(f"Exact Match: {metrics['exact_match_accuracy']:.2f}%")
    print(f"BLEU: {metrics['bleu_score_mean']:.4f} (±{metrics['bleu_score_std']:.4f})")
    print(f"CER: {metrics['cer_mean']:.4f} (±{metrics['cer_std']:.4f})")
    print(f"WER: {metrics['wer_mean']:.4f} (±{metrics['wer_std']:.4f})")
    print(
        f"Edit Distance: {metrics['edit_distance_mean']:.2f} (±{metrics['edit_distance_std']:.2f})"
    )
    print("=" * 80)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/eval_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    volume.commit()
    print(f"\n✓ Results saved to: {results_file}")

    return {
        "status": "completed",
        "metrics": metrics,
        "results_file": results_file,
    }


# ==============================================================================
# INFERENCE
# ==============================================================================


@app.function(
    image=image,
    gpu="L40s",
    volumes={"/data": volume},
    timeout=30 * 60,
    secrets=[huggingface_secret],
)
def inference_finetuned(
    image_path: str = None,
    image_data: str = None,
    model_path=f"{CHECKPOINTS_DIR}/llada-latex-ocr",
    max_new_tokens: int = 512,
):
    """
    Run inference with fine-tuned LLaDA-V model.

    Automatically detects model type (LoRA checkpoint, merged model, or HuggingFace repo).

    Args:
        image_path: Path to image file
        image_data: Base64-encoded image data
        model_path: Path to model (LoRA checkpoint, merged model, or HuggingFace repo ID)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with generated LaTeX and metadata
    """
    import sys

    sys.path.insert(0, "/root")

    import torch
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("FINE-TUNED LLADA-V INFERENCE")
    print("=" * 80)
    print(f"Model: {model_path}")
    print()

    # Load model with smart detection
    print("[1/3] Loading model...")

    # Auto-detect model type and load accordingly
    model_type = None
    base_model_path = f"{MODELS_DIR}/LLaDA-V"

    if os.path.exists(model_path):
        # Local path - check if it's a directory with checkpoints
        if os.path.isdir(model_path):
            checkpoint_dirs = [
                d
                for d in os.listdir(model_path)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(model_path, d))
            ]
            if checkpoint_dirs:
                # Has checkpoint subdirectories - use latest
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = checkpoint_dirs[-1]
                model_path = os.path.join(model_path, latest_checkpoint)
                print(f"Auto-selected latest checkpoint: {latest_checkpoint}")

        # Check if it's a LoRA checkpoint
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            model_type = "lora"
            print("Detected LoRA checkpoint")
        else:
            model_type = "merged"
            print("Detected merged/full model")
    else:
        # Not a local path - assume HuggingFace Hub repo
        model_type = "huggingface"
        print(f"Loading from HuggingFace Hub: {model_path}")

    # Load model based on type
    if model_type == "lora":
        # Load base model + LoRA adapters
        # Note: model_name must include "lora" to trigger LoRA loading path
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,  # LoRA checkpoint path
            base_model_path,  # Base model path
            "lora-llava-llada",  # Must include "lora" in name
            attn_implementation="sdpa",
            device_map=device,
        )
    else:
        # Load merged model or HuggingFace model
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            "llava_llada",
            attn_implementation="sdpa",
            device_map=device,
        )

    model.eval()
    print(f"✓ Model loaded ({model_type}) on {device}")

    # Load image
    print("\n[2/3] Loading image...")
    if image_data and image_data.startswith("data:image"):
        _, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        print("✓ Loaded from base64 data")
    elif image_path:
        image = PILImage.open(image_path).convert("RGB")
        print(f"✓ Loaded from {image_path}")
    else:
        raise ValueError("Must provide either image_path or image_data")

    print(f"  Image size: {image.size}")

    # Prepare input
    print("\n[3/3] Generating LaTeX...")
    text = "<image>\nConvert this mathematical expression to LaTeX code."
    input_ids = (
        tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    image_tensor = process_images([image], image_processor, model.config).to(
        dtype=torch.float16, device=device
    )

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )[0]

    latex = tokenizer.decode(
        output_ids[input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    print("=" * 80)
    print("RESULT:")
    print("-" * 80)
    print(latex)
    print("=" * 80)

    return {
        "status": "success",
        "latex": latex,
        "model": model_path,
        "model_type": model_type,
        "image_size": image.size,
    }


"""
LLaDA-V Fine-tuning Pipeline
============================

Complete workflow for fine-tuning LLaDA-V on LaTeX OCR dataset.

Configuration:
--------------
Edit OUTPUT_BASE_DIR at the top of this file to organize your experiments.
Default paths:
- Models: /data/models
- Datasets: /data/datasets
- Checkpoints: /data/checkpoints

To separate experiments, change OUTPUT_BASE_DIR (e.g., "/data/experiment1")

Step 1: Download and prepare dataset
-------------------------------------
# HuggingFace Arrow format (RECOMMENDED - 2-5x faster loading, no corrupted images)
modal run modal_finetune_llada.py::download_and_prepare_dataset --save-format arrow
modal run modal_finetune_llada.py::download_and_prepare_dataset --split validation --save-format arrow
modal run modal_finetune_llada.py::download_and_prepare_dataset --max-samples 100 --save-format arrow

# Legacy JSON format (backward compatibility)
modal run modal_finetune_llada.py::download_and_prepare_dataset --save-format json
modal run modal_finetune_llada.py::download_and_prepare_dataset --split validation --save-format json
modal run modal_finetune_llada.py::download_and_prepare_dataset --max-samples 10000 --save-format json

Step 2: Download pre-trained model
-----------------------------------
modal run modal_finetune_llada.py::download_model

Step 3: Train model
-------------------
# LoRA fine-tuning with Arrow format (RECOMMENDED)
modal run modal_finetune_llada.py::train_llada \
    --use-lora True \
    --use-hf-dataset True

# LoRA fine-tuning with JSON format (legacy)
modal run modal_finetune_llada.py::train_llada \
    --use-lora True \
    --use-hf-dataset False \
    --dataset-path /data/datasets/latex_ocr_dataset_train/dataset.json

# Full fine-tuning with Arrow format
modal run modal_finetune_llada.py::train_llada \
    --use-lora False \
    --use-hf-dataset True

# With DeepSpeed (multi-GPU, JSON format)
modal run modal_finetune_llada.py::train_llada \
    --use-lora True \
    --use-hf-dataset False \
    --dataset-path /data/datasets/latex_ocr_dataset_train/dataset.json \
    --deepspeed-config "zero3"

Step 4a: Process LoRA checkpoint (if using LoRA)
-------------------------------------------------
# Default: uses training alpha from checkpoint (e.g., checkpoint-1000-merged-alpha128)
modal run modal_finetune_llada.py::merge_lora_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr/checkpoint-1000

# Custom alpha: 2x rank (for rank=64, alpha=128)
modal run modal_finetune_llada.py::merge_lora_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr/checkpoint-1000 \
    --merge-alpha 128

# Custom alpha: 4x rank (for rank=64, alpha=256)
modal run modal_finetune_llada.py::merge_lora_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr/checkpoint-1000 \
    --merge-alpha 256

# Custom output path (overrides automatic alpha suffix)
modal run modal_finetune_llada.py::merge_lora_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr/checkpoint-1000 \
    --output-path /data/merged_models/my-model

# Push to HuggingFace Hub with custom alpha
modal run modal_finetune_llada.py::merge_lora_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr/checkpoint-1000 \
    --merge-alpha 128 \
    --huggingface-repo-id "username/model-name" \
    --private True

Step 4b: Process full fine-tuned checkpoint (if using full fine-tuning)
------------------------------------------------------------------------
# Default: checkpoint-7500-processed (removes optimizer.pt, saves ~15GB)
modal run modal_finetune_llada.py::process_full_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr-full/checkpoint-7500

# Custom output path
modal run modal_finetune_llada.py::process_full_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr-full/checkpoint-7500 \
    --output-path /data/models/llada-latex-final

# Push to HuggingFace Hub
modal run modal_finetune_llada.py::process_full_checkpoint \
    --checkpoint-path /data/checkpoints/llada-latex-ocr-full/checkpoint-7500 \
    --huggingface-repo-id "username/llada-latex-ocr" \
    --private True

Step 5: Evaluate model
----------------------
# Evaluate merged model (RECOMMENDED)
# Note: Path includes alpha value (e.g., checkpoint-1000-merged-alpha128)
modal run modal_finetune_llada.py::evaluate_model \
    --model-path /data/checkpoints/llada-latex-ocr/checkpoint-1000-merged-alpha128

# Evaluate from HuggingFace Hub
modal run modal_finetune_llada.py::evaluate_model \
    --model-path "username/model-name"

# Evaluate base model (for comparison)
modal run modal_finetune_llada.py::evaluate_model \
    --model-path /data/models/LLaDA-V

Step 6: Test inference
----------------------
# Use merged model (includes alpha in path)
modal run modal_finetune_llada.py::inference_finetuned \
    --image-path test.png \
    --model-path /data/checkpoints/llada-latex-ocr/checkpoint-1000-merged-alpha128

# Use HuggingFace Hub model
modal run modal_finetune_llada.py::inference_finetuned \
    --image-path test.png \
    --model-path "username/model-name"
"""
