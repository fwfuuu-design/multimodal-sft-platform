# Configuration examples

[中文说明](README_zh.md)

This directory provides configuration examples for common platform tasks.

- `text_lora_sft.yaml`: text SFT with LoRA
- `image_qlora_sft.yaml`: image-text SFT with 4-bit bitsandbytes QLoRA
- `adapter_merge.yaml`: merge an Adapter into an unquantized base model

The model and output paths in these files are placeholders. Update them for the target environment or validate a training configuration with:

```bash
sft-train train --dry-run examples/text_lora_sft.yaml
```
