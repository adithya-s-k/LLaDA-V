# LLaDA-V Modal Inference Scripts

This directory contains Modal deployment scripts for running inference with LLaDA-V, a diffusion-based vision-language model optimized for document understanding and text extraction.

## Quick Start

```bash
# Navigate to LLaDA-V directory
cd /Users/adithyaskolavi/projects/eagle/LLaDA-V

# Run basic inference
modal run modal_scripts/modal_inference_llada.py --image-path your_document.png
```

## Scripts

### `modal_inference_llada.py`

Main inference script for OCR and text extraction from document images.

**Features:**
- ( Three acceleration modes: Fast-dLLM (~6s), Standard Cache (~25s), Baseline (~60s)
- =Ä Optimized for document text extraction
- = Downloads model automatically from HuggingFace (GSAI-ML/LLaDA-V)
- =¼ Supports image paths, URLs, and base64 data
- <¯ Pre-configured OCR prompt templates

## Usage Examples

### Basic Usage

```bash
# Simple text extraction with default settings (Fast-dLLM)
modal run modal_scripts/modal_inference_llada.py \
  --image-path document.png
```

### Custom Prompts

```bash
# Extract specific information
modal run modal_scripts/modal_inference_llada.py \
  --image-path invoice.jpg \
  --prompt "Extract all invoice line items with prices and quantities"

# Use pre-configured prompt templates
modal run modal_scripts/modal_inference_llada.py \
  --image-path form.png \
  --prompt-type forms
```

Available prompt templates:
- `default`: General text extraction with formatting
- `structured`: Maintain layout with headers and sections
- `tables`: Extract tables with structure preservation
- `forms`: Extract form fields and values
- `handwritten`: OCR for handwritten text

### Acceleration Modes

```bash
# Fast mode (default) - Fast-dLLM hooks, ~6-10s for 128 tokens
modal run modal_scripts/modal_inference_llada.py \
  --image-path doc.png \
  --acceleration fast

# Standard mode - dLLM-Cache, ~25-30s for 128 tokens
modal run modal_scripts/modal_inference_llada.py \
  --image-path doc.png \
  --acceleration standard

# No acceleration - Baseline diffusion, ~60s for 128 tokens
modal run modal_scripts/modal_inference_llada.py \
  --image-path doc.png \
  --acceleration none
```

### Image Sources

```bash
# From local file path
modal run modal_scripts/modal_inference_llada.py \
  --image-path /path/to/document.png

# From URL
modal run modal_scripts/modal_inference_llada.py \
  --image-url https://example.com/document.png
```

### Generation Parameters

```bash
# Longer generation for detailed documents
modal run modal_scripts/modal_inference_llada.py \
  --image-path complex_doc.png \
  --gen-length 1024

# Custom diffusion steps
modal run modal_scripts/modal_inference_llada.py \
  --image-path doc.png \
  --steps 256 \
  --gen-length 512
```

## Performance Benchmarks

| Acceleration Mode | Time (128 tokens) | Time (512 tokens) | Best For |
|-------------------|-------------------|-------------------|----------|
| Fast (Fast-dLLM) | ~6-10s | ~15-25s | Production, real-time |
| Standard (dLLM-Cache) | ~25-30s | ~60-80s | Balanced quality/speed |
| None (Baseline) | ~60s | ~180s | Development, testing |

**Notes:**
- Times measured on L40s GPU
- Model load time: ~20-30s (with volume caching)
- First run takes longer due to model download

## Configuration

### Required Secrets

The script requires a HuggingFace token stored in Modal secrets:

```bash
# Set up Modal secret (one-time setup)
modal secret create adithya-hf-wandb \
  HUGGINGFACE_TOKEN=your_hf_token_here
```

### GPU Options

The default GPU is L40s. You can modify this in the script:

```python
@app.function(
    gpu="L40s",  # Options: "L40s", "A100-40GB", "A100-80GB", "H100"
    # ...
)
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image-path` | str | None | Path to local image file |
| `--image-url` | str | None | URL to download image from |
| `--prompt` | str | None | Custom prompt (default: OCR extraction) |
| `--prompt-type` | str | None | Pre-configured prompt template |
| `--acceleration` | str | "fast" | Acceleration mode: fast, standard, none |
| `--steps` | int | 128 | Diffusion steps |
| `--gen-length` | int | 512 | Maximum tokens to generate |
| `--block-length` | int | 128 | Block length for generation |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce generation length
modal run modal_scripts/modal_inference_llada.py \
  --image-path doc.png \
  --gen-length 256

# Or use a larger GPU (edit the script)
```

### Model Download Issues

```bash
# Ensure HuggingFace token is set correctly
modal secret list

# Check token has access to GSAI-ML/LLaDA-V model
```

### Slow First Run

The first run downloads the model (~8GB) which takes 5-10 minutes. Subsequent runs use cached model and start in ~30s.

## Architecture Details

### Model: LLaDA-V

- **Base**: Diffusion-based language model with visual instruction tuning
- **Vision Encoder**: SigLIP2 (siglip2-so400m-patch14-384)
- **Model ID**: GSAI-ML/LLaDA-V
- **Context**: 128 tokens with diffusion process
- **Tokenizer**: Llama 3 based

### Acceleration Mechanisms

**Fast-dLLM:**
- Implements distributed LLM caching
- Hooks into attention layers for cache management
- Reduces generation latency from 60s to 6s on A100

**dLLM-Cache:**
- Alternative caching mechanism
- Configurable refresh intervals
- Balances speed and quality

## Related Resources

- [LLaDA-V Paper](https://arxiv.org/abs/2505.16933)
- [LLaDA-V Model on HuggingFace](https://huggingface.co/GSAI-ML/LLaDA-V)
- [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
- [Modal Documentation](https://modal.com/docs)

## License

This script follows the Apache 2.0 license of the LLaDA-V project.
