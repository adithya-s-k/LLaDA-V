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
):
    """
    Download and prepare LaTeX OCR dataset in LLaDA-V conversation format.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split ("train" or "validation")
        max_samples: Optional limit for testing (None for all samples)

    Returns:
        Dict with status, dataset path, and sample count
    """
    from datasets import load_dataset
    from PIL import Image
    from io import BytesIO

    target_dataset_path = f"/data/latex_ocr_dataset_{split}"

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

    # Step 1: Filter valid samples
    print("Step 1/2: Filtering valid samples...")
    valid_dataset = dataset.filter(is_valid_sample, desc="Filtering")
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

    # Save dataset in JSON format for LLaDA-V training
    import json
    import os

    os.makedirs(target_dataset_path, exist_ok=True)
    images_dir = os.path.join(target_dataset_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Convert dataset to JSON with saved images
    json_data = []
    for sample in filtered_dataset:
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

    success_rate = len(filtered_dataset) / len(dataset) * 100
    print(
        f"✓ Saved {len(filtered_dataset)}/{len(dataset)} samples ({success_rate:.1f}%)"
    )
    print(f"  JSON file: {json_path}")
    print(f"  Images directory: {images_dir}")

    return {
        "status": "completed",
        "dataset_path": json_path,  # Return JSON file path, not directory
        "samples": len(filtered_dataset),
        "success_rate": f"{success_rate:.1f}%",
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

    os.makedirs("/data/models", exist_ok=True)

    model_path = "/data/models/LLaDA-V"
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


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": volume},
    timeout=24 * HOURS,
    secrets=[huggingface_secret, Secret.from_dotenv()],
)
def train_llada(
    dataset_path="/data/latex_ocr_dataset_train/dataset.json",
    model_path="/data/models/LLaDA-V",
    output_dir="/data/checkpoints/llada-latex-ocr",
    # Training mode
    use_lora=False,  # True for LoRA, False for full fine-tuning
    # Tunable components (for LoRA: freeze vision tower, train adapter + LLM)
    mm_tunable_parts="mm_mlp_adapter,mm_language_model",
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.0,
    # LoRA-specific parameters
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    # DeepSpeed configuration (optional - only use for multi-GPU training)
    deepspeed_config=None,  # None = no DeepSpeed, "zero2" or "zero3" = enable DeepSpeed
    # Other settings
    save_steps=500,
    logging_steps=10,
    logging_nan_inf_filter=False,  # Log all values including NaN/Inf for debugging
):
    """
    Train LLaDA-V on LaTeX OCR dataset.

    Supports both LoRA and full fine-tuning modes with configurable components.

    Args:
        dataset_path: Path to prepared dataset
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
        print(f"Effective batch size maintained at {per_device_train_batch_size * gradient_accumulation_steps} "
              f"via gradient accumulation")

    print("=" * 80)
    print("LLADA-V FINE-TUNING ON LATEX OCR")
    print("=" * 80)
    print(f"Training mode: {'LoRA' if use_lora else 'Full fine-tuning'}")
    print(f"DeepSpeed: {'Enabled (' + deepspeed_config + ')' if deepspeed_config else 'Disabled (single-GPU)'}")
    print(f"Tunable components: {mm_tunable_parts}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Batch size: {per_device_train_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
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

    # Build training command based on llada_v_finetune.sh
    # Extract image folder from dataset path (parent directory of JSON file)
    import os as os_module

    dataset_dir = os_module.path.dirname(dataset_path)

    command = [
        "python",
        "/root/llava/train/train_mem.py",
        # Model and data configuration
        "--model_name_or_path",
        model_path,
        "--version",
        "llava_llada",
        "--data_path",
        dataset_path,
        "--image_folder",
        dataset_dir,
        "--vision_tower",
        model_path,  # Vision tower is integrated in LLaDA-V
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

    # Add DeepSpeed flag if enabled
    if deepspeed_config:
        deepspeed_config_path = f"/root/scripts/{deepspeed_config}.json"
        if not os.path.exists(deepspeed_config_path):
            raise FileNotFoundError(
                f"DeepSpeed config not found: {deepspeed_config_path}\n"
                f"Available configs: zero2.json, zero3.json, zero2_offload.json, zero3_offload.json"
            )
        command.extend([
            "--deepspeed",
            deepspeed_config_path,
        ])
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
    checkpoint_path="/data/checkpoints/llada-latex-ocr",
    dataset_path="/data/latex_ocr_dataset_validation/dataset.json",
    output_dir="/data/evaluation_results",
    max_samples: int = None,
    max_new_tokens: int = 512,
    batch_size: int = 4,  # Number of samples to process at once
):
    """
    Evaluate fine-tuned LLaDA-V model on validation set.

    Calculates metrics:
    - Exact Match Accuracy
    - BLEU Score
    - Character Error Rate (CER)
    - Word Error Rate (WER)
    - Edit Distance (Levenshtein)

    Args:
        checkpoint_path: Path to fine-tuned checkpoint
        dataset_path: Path to validation dataset
        output_dir: Directory to save evaluation results
        max_samples: Maximum samples to evaluate (None for all)
        max_new_tokens: Maximum tokens to generate
        batch_size: Number of samples to process at once (default=4 for speed)

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
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print()

    # Load model
    print("[1/3] Loading model...")

    # Find latest checkpoint if directory contains multiple
    if os.path.exists(checkpoint_path):
        checkpoint_dirs = [
            d
            for d in os.listdir(checkpoint_path)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(checkpoint_path, d))
        ]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoint_dirs[-1]
            checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
            print(f"Using latest checkpoint: {latest_checkpoint}")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        checkpoint_path,
        None,  # No model_base needed for fine-tuned checkpoint
        "llava_llada",
        attn_implementation="sdpa",  # Use SDPA instead of Flash Attention 2
        device_map=device,
    )
    model.eval()

    print("✓ Model loaded")
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
        "checkpoint": checkpoint_path,
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
    checkpoint_path="/data/checkpoints/llada-latex-ocr",
    max_new_tokens: int = 512,
):
    """
    Run inference with fine-tuned LLaDA-V model.

    Args:
        image_path: Path to image file
        image_data: Base64-encoded image data
        checkpoint_path: Path to fine-tuned checkpoint
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

    # Load model
    print("[1/3] Loading model...")

    # Find latest checkpoint if directory contains multiple
    if os.path.exists(checkpoint_path):
        checkpoint_dirs = [
            d
            for d in os.listdir(checkpoint_path)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(checkpoint_path, d))
        ]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoint_dirs[-1]
            checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
            print(f"Using latest checkpoint: {latest_checkpoint}")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        checkpoint_path,
        None,
        "llava_llada",
        attn_implementation="sdpa",  # Use SDPA instead of Flash Attention 2
        device_map=device,
    )
    model.eval()
    print(f"✓ Model loaded on {device}")

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
        "checkpoint": checkpoint_path,
        "image_size": image.size,
    }


# ==============================================================================
# CLI ENTRYPOINTS
# ==============================================================================


@app.local_entrypoint()
def main():
    """
    Main CLI entrypoint with usage instructions.
    """
    print("""
LLaDA-V Fine-tuning Pipeline
============================

Step 1: Download and prepare dataset
  # Full training dataset
  modal run modal_finetune_llada.py::download_and_prepare_dataset

  # Validation dataset
  modal run modal_finetune_llada.py::download_and_prepare_dataset --split validation

  # Limited samples for testing (e.g., 10000 samples)
  modal run modal_finetune_llada.py::download_and_prepare_dataset --max-samples 10000
  modal run modal_finetune_llada.py::download_and_prepare_dataset --split validation --max-samples 1000

Step 2: Download pre-trained model
  modal run modal_finetune_llada.py::download_model

Step 3: Train model
  # LoRA fine-tuning (faster, smaller checkpoints)
  modal run modal_finetune_llada.py::train_llada --use-lora True

  # Full fine-tuning (best performance)
  modal run modal_finetune_llada.py::train_llada --use-lora False

  # Custom tunable components
  modal run modal_finetune_llada.py::train_llada --mm-tunable-parts "mm_mlp_adapter"

Step 4: Evaluate model
  modal run modal_finetune_llada.py::evaluate_model
  
  modal run modal_finetune_llada.py::evaluate_model --checkpoint-path /data/models/LLaDA-V

Step 5: Test inference
  modal run modal_finetune_llada.py::inference_finetuned --image-path test.png

For more options, use --help with any function.
    """)
