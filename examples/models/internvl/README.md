
# InternViT Model

This directory contains the implementation and conversion scripts for the InternViT Vision Transformer model, used in the SAfe autonomous driving project.

## Files

- `configuration_intern_vit.py`: Configuration class for the InternViT model.
- `configuration_internvl_chat.py`: Configuration for InternVL chat model.
- `convert_checkpoint.py`: Script to convert pretrained InternViT model to ONNX and TensorRT formats.
- `modeling_intern_vit.py`: PyTorch implementation of the InternViT model with FlashAttention support.
- `vit_onnx_trt.py`: Wrapper class for InternViT model with pixel shuffle functionality.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- TensorRT
- ONNX
- PIL (Pillow)
- torchvision
- einops
- timm
- flash-attn (optional, for FlashAttention support)

## Usage

To convert a pretrained InternViT model to ONNX and TensorRT formats, run the following command:

```bash
uv run python examples/models/internvl/vit_onnx_trt.py \
    --pretrained_model_path "owl10/ReCogDrive-VLM-2B" \
    --onnxFile ./examples/models/internvl/models/vision_projector.onnx \
    --planFile ./models/vision_projector_bf16.plan \
    --dtype bfloat16 \
    --minBS 1 \
    --optBS 9 \
    --maxBS 16 \
    --image_url ./examples/models/internvl/pics/demo.jpeg
```

### Arguments

- `--pretrained_model_path`: Path or Hugging Face model ID of the pretrained model
- `--onnxFile`: Output path for the ONNX file
- `--planFile`: Output path for the TensorRT plan file
- `--image_url`: Path to a demo image for testing

## Model Architecture

InternViT is a Vision Transformer model that processes images and extracts features. It supports:

- Patch-based image processing
- Multi-head self-attention with optional FlashAttention
- MLP layers for feature transformation
- Pixel shuffle for upsampling

## License

This implementation is based on the InternVL model and follows the MIT License.

