"""
LLaDA-V Modal Inference Script
================================

Run inference with LLaDA-V (diffusion-based vision-language model) for OCR/text extraction.

Features:
- Three acceleration modes: fast (Fast-dLLM), standard (dLLM-Cache), none (baseline)
- Downloads model from HuggingFace Hub (GSAI-ML/LLaDA-V)
- Optimized for document text extraction
- Support for image paths, URLs, and base64 data

Usage:
    # Basic usage with Fast-dLLM (default)
    modal run modal_inference_llada.py --image-path document.png

    # With custom prompt
    modal run modal_inference_llada.py --image-path form.jpg --prompt "Extract form fields"

    # Different acceleration mode
    modal run modal_inference_llada.py --image-path page.png --acceleration standard

    # From URL
    modal run modal_inference_llada.py --image-url https://example.com/doc.png

    # Longer generation for detailed documents
    modal run modal_inference_llada.py --image-path doc.png --gen-length 1024
"""

from modal import App, Image, Volume, Secret
from pathlib import Path

# ==============================================================================
# MODAL SETUP
# ==============================================================================

app = App("llada-v-inference")
volume = Volume.from_name("llada-v-vol", create_if_missing=True)

# CUDA configuration (matching Eagle/Bunny projects)
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Build Modal image with dependencies
image = (
    Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(
        [
            "torch",
            "torchvision",
            "transformers==4.40.0",  # Pin to match LLaDA-V requirements
            "accelerate==0.29.3",
            "pillow",
            "timm==0.9.16",
            "einops==0.6.1",
            "safetensors==0.4.3",
            "huggingface_hub==0.22.2",
            "requests",
        ]
    )
    .env(
        {
            # "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",
        }
    )
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "train"),
        remote_path="/root",
        copy=True,
        ignore=[".git", "__pycache__", "*.pyc", ".DS_Store", "data/", "checkpoints/"],
    )
)

huggingface_secret = Secret.from_name("adithya-hf-wandb")

# ==============================================================================
# OCR PROMPT TEMPLATES
# ==============================================================================

OCR_PROMPTS = {
    "default": "Extract all text from this document image. Preserve the structure and formatting as much as possible.",
    "structured": "Extract all text from this document while maintaining the original layout, including headers, paragraphs, and sections.",
    "tables": "Extract all text from this document, paying special attention to table structures. Preserve row and column relationships.",
    "forms": "Extract all text from this form, including field labels and their corresponding values.",
    "handwritten": "Extract all handwritten text from this document image as accurately as possible.",
}

# ==============================================================================
# MAIN INFERENCE FUNCTION
# ==============================================================================


@app.function(
    image=image,
    gpu="L40s",
    secrets=[huggingface_secret],
    volumes={"/data": volume},
    timeout=2 * 60 * 60,  # 2 hours
)
def inference_llada(
    image_path: str = None,
    image_data: str = None,
    image_url: str = None,
    prompt: str = None,
    acceleration: str = "fast",  # "fast", "standard", "none"
    steps: int = 128,
    gen_length: int = 512,  # Longer for document text
    block_length: int = 128,
):
    """
    Run LLaDA-V inference for text extraction from document images.

    Args:
        image_path: Path to local image file
        image_data: Base64-encoded image data
        image_url: URL to download image from
        prompt: Custom prompt (defaults to OCR extraction)
        acceleration: Acceleration mode - "fast" (Fast-dLLM), "standard" (dLLM-Cache), or "none"
        steps: Diffusion steps (default 128)
        gen_length: Maximum tokens to generate (default 512)
        block_length: Block length for generation (default 128)

    Returns:
        Dict with extracted_text, generation_time, and other metadata
    """
    import sys

    sys.path.insert(0, "/root")  # Critical for llava imports

    import os
    import torch
    import time
    import warnings
    import copy
    from io import BytesIO
    import base64
    import requests
    from PIL import Image as PILImage
    from dataclasses import asdict

    # Import LLaDA-V modules
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.cache import dLLMCache, dLLMCacheConfig
    from llava.hooks import register_cache_LLaDA_V
    from llava.hooks.fast_dllm_hook import register_fast_dllm_hook

    warnings.filterwarnings("ignore")

    # Set environment variables
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")
    os.environ["HF_TOKEN"] = os.environ.get("HUGGINGFACE_TOKEN", "")

    print("=" * 80)
    print("LLaDA-V INFERENCE - OCR/Text Extraction")
    print("=" * 80)
    print(f"Acceleration mode: {acceleration}")
    print(f"Generation length: {gen_length} tokens")
    print()

    # =========================================================================
    # STEP 1: Load Model
    # =========================================================================
    print("[1/5] Loading LLaDA-V model...")
    print("  Note: Vision tower (SigLIP) weights are included in LLaDA-V checkpoint")
    print("  No separate download needed - all weights load from GSAI-ML/LLaDA-V")
    print()

    device = "cuda:0"
    pretrained = "GSAI-ML/LLaDA-V"
    model_name = "llava_llada"  # Critical: triggers correct model loading

    try:
        # This loads:
        # - LLaDA language model (diffusion-based)
        # - MM projector (vision-to-text alignment)
        # - SigLIP vision tower (already fine-tuned, included in checkpoint)
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained,
            None,  # No base model path needed
            model_name,
            attn_implementation="sdpa",
            device_map=device,
        )
        model.eval()
        print("\n✓ Model loaded successfully (including vision tower)")
        print(f"  Device: {device}")
        print(f"  Max length: {max_length}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Tip: Ensure HUGGINGFACE_TOKEN is set in Modal secrets")
        raise

    # =========================================================================
    # STEP 2: Configure Acceleration
    # =========================================================================
    print(f"\n[2/5] Configuring acceleration mode: {acceleration}...")

    if acceleration == "fast":
        register_fast_dllm_hook(model)
        print("✓ Fast-dLLM hooks registered (~6-10s per 128 tokens)")

    elif acceleration == "standard":
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=25,
                    gen_interval_steps=7,
                    transfer_ratio=0.25,
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        print("✓ Standard dLLM-Cache configured (~25-30s per 128 tokens)")

    else:  # acceleration == "none"
        print("✓ No acceleration (baseline diffusion, ~60s per 128 tokens)")

    # =========================================================================
    # STEP 3: Load and Process Image
    # =========================================================================
    print("\n[3/5] Loading image...")

    try:
        if image_path:
            image = PILImage.open(image_path).convert("RGB")
            print(f"✓ Loaded from path: {image_path}")
        elif image_data:
            # Handle base64 data
            if image_data.startswith("data:image"):
                _, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                image_bytes = base64.b64decode(image_data)
            image = PILImage.open(BytesIO(image_bytes)).convert("RGB")
            print("✓ Loaded from base64 data")
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image = PILImage.open(BytesIO(response.content)).convert("RGB")
            print(f"✓ Loaded from URL: {image_url}")
        else:
            # No image provided - create a synthetic test document
            print("✓ No image provided, creating synthetic test document...")
            from PIL import ImageDraw, ImageFont

            width, height = 800, 1000
            image = PILImage.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)

            # Use default font
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                heading_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except (OSError, IOError):
                title_font = ImageFont.load_default()
                heading_font = ImageFont.load_default()
                body_font = ImageFont.load_default()

            y = 50

            # Title
            draw.text((50, y), "Invoice #12345", font=title_font, fill='black')
            y += 60

            # Info
            draw.text((50, y), "Date: November 25, 2025", font=body_font, fill='black')
            y += 30
            draw.text((50, y), "Customer: John Smith", font=body_font, fill='black')
            y += 30
            draw.text((50, y), "Address: 123 Main Street, San Francisco, CA 94102", font=body_font, fill='black')
            y += 50

            # Section
            draw.text((50, y), "Items:", font=heading_font, fill='black')
            y += 40

            # Table header
            draw.text((50, y), "Item", font=body_font, fill='black')
            draw.text((300, y), "Qty", font=body_font, fill='black')
            draw.text((400, y), "Price", font=body_font, fill='black')
            draw.text((550, y), "Total", font=body_font, fill='black')
            y += 30
            draw.line([(50, y), (700, y)], fill='black', width=1)
            y += 10

            # Items
            items = [
                ("Product A", "2", "$50.00", "$100.00"),
                ("Product B", "1", "$75.00", "$75.00"),
                ("Product C", "3", "$25.00", "$75.00"),
            ]

            for item, qty, price, total in items:
                draw.text((50, y), item, font=body_font, fill='black')
                draw.text((300, y), qty, font=body_font, fill='black')
                draw.text((400, y), price, font=body_font, fill='black')
                draw.text((550, y), total, font=body_font, fill='black')
                y += 30

            # Total section
            draw.line([(50, y), (700, y)], fill='black', width=1)
            y += 20
            draw.text((400, y), "Subtotal:", font=body_font, fill='black')
            draw.text((550, y), "$250.00", font=body_font, fill='black')
            y += 30
            draw.text((400, y), "Tax (8%):", font=body_font, fill='black')
            draw.text((550, y), "$20.00", font=body_font, fill='black')
            y += 30
            draw.line([(400, y), (700, y)], fill='black', width=2)
            y += 10
            draw.text((400, y), "Total:", font=heading_font, fill='black')
            draw.text((550, y), "$270.00", font=heading_font, fill='black')
            y += 60

            # Footer
            draw.text((50, y), "Payment Terms:", font=heading_font, fill='black')
            y += 35
            draw.text((50, y), "Payment due within 30 days.", font=body_font, fill='black')
            y += 25
            draw.text((50, y), "Make checks payable to: Acme Corporation", font=body_font, fill='black')
            y += 25
            draw.text((50, y), "Thank you for your business!", font=body_font, fill='black')

            print("  Created synthetic invoice document")

        print(f"  Image size: {image.size[0]}x{image.size[1]}")

    except Exception as e:
        print(f"✗ Error loading image: {e}")
        raise ValueError(f"Could not load image from provided source: {e}")

    # Process image for model
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [
        _image.to(dtype=torch.float16, device=device) for _image in image_tensor
    ]
    image_sizes = [image.size]  # (width, height) tuple

    # =========================================================================
    # STEP 4: Prepare Prompt
    # =========================================================================
    print("\n[4/5] Preparing prompt...")

    # Use default OCR prompt if none provided
    if prompt is None:
        prompt = OCR_PROMPTS["default"]
        print("✓ Using default OCR prompt")
    else:
        print("✓ Using custom prompt")

    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    # Build conversation using llava_llada template
    conv_template = "llava_llada"
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # Tokenize
    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )

    print(f"  Input tokens: {input_ids.shape[1]}")

    # =========================================================================
    # STEP 5: Generate Response
    # =========================================================================
    print("\n[5/5] Generating response...")
    print(f"  Steps: {steps}, Gen length: {gen_length}, Block: {block_length}")

    start_time = time.time()

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                tokenizer=tokenizer,
                stopping_criteria=["<|eot_id|>"],  # Llama 3 end-of-turn token
                prefix_refresh_interval=32,
                threshold=1,
            )

    except torch.cuda.OutOfMemoryError:
        print("✗ CUDA Out of Memory Error")
        print(f"  Try: --gen-length {gen_length // 2} or use a larger GPU")
        raise
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        raise

    end_time = time.time()
    generation_time = end_time - start_time

    # Decode output
    text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    extracted_text = text_outputs[0].strip()

    print("✓ Generation complete")
    print(f"  Generation time: {generation_time:.2f}s")
    print(f"  Output length: {len(extracted_text)} characters")
    print(f"  Output tokens: {output_ids.shape[1]}")

    # =========================================================================
    # Return Results
    # =========================================================================
    return {
        "status": "success",
        "extracted_text": extracted_text,
        "generation_time": generation_time,
        "acceleration_mode": acceleration,
        "image_size": image.size,
        "input_tokens": input_ids.shape[1],
        "output_tokens": output_ids.shape[1],
        "steps": steps,
        "gen_length": gen_length,
    }


# ==============================================================================
# LOCAL ENTRYPOINT (CLI)
# ==============================================================================


@app.local_entrypoint()
def main(
    image_path: str = None,
    image_url: str = None,
    prompt: str = None,
    prompt_type: str = None,
    acceleration: str = "fast",
    steps: int = 128,
    gen_length: int = 512,
    block_length: int = 128,
):
    """
    Run LLaDA-V inference from command line.

    Examples:
        # Test with synthetic document (no image needed)
        modal run modal_inference_llada.py

        # With your own image
        modal run modal_inference_llada.py --image-path document.png

        # Custom prompt
        modal run modal_inference_llada.py --image-path form.jpg --prompt "Extract form fields"

        # Different acceleration
        modal run modal_inference_llada.py --acceleration standard

        # From URL
        modal run modal_inference_llada.py --image-url https://example.com/doc.png

        # Longer generation
        modal run modal_inference_llada.py --gen-length 1024

        # Pre-configured prompt template
        modal run modal_inference_llada.py --prompt-type tables
    """

    # No validation needed - if no image is provided, we'll create a synthetic one
    if not image_path and not image_url:
        print("Note: No image provided - will create a synthetic test document")
        print()

    # Validate acceleration mode
    if acceleration not in ["fast", "standard", "none"]:
        print(f"Error: Invalid acceleration mode '{acceleration}'")
        print("Valid options: fast, standard, none")
        return

    # Use pre-configured prompt template if specified
    if prompt_type:
        if prompt_type in OCR_PROMPTS:
            prompt = OCR_PROMPTS[prompt_type]
            print(f"Using '{prompt_type}' prompt template")
        else:
            print(f"Error: Unknown prompt type '{prompt_type}'")
            print(f"Valid options: {', '.join(OCR_PROMPTS.keys())}")
            return

    print("\n" + "=" * 80)
    print("LLADA-V INFERENCE")
    print("=" * 80)
    print(f"Image: {image_path or image_url}")
    print(f"Acceleration: {acceleration}")
    print(f"Generation length: {gen_length} tokens")
    if prompt:
        print(f"Custom prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print("=" * 80 + "\n")

    # Run inference
    result = inference_llada.remote(
        image_path=image_path,
        image_url=image_url,
        prompt=prompt,
        acceleration=acceleration,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Acceleration mode: {result['acceleration_mode']}")
    print(f"Image size: {result['image_size']}")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print("=" * 80)
    print("\nEXTRACTED TEXT:")
    print("-" * 80)
    print(result["extracted_text"])
    print("-" * 80)

    return result
