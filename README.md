# Multimodal SFT Platform

[中文说明](README_zh.md)

Multimodal SFT Platform provides supervised fine-tuning workflows for text and image-text datasets.

## Features

- LoRA and 4-bit bitsandbytes QLoRA training
- Evaluation, prediction, image-text chat, and Adapter export
- Authenticated API and WebUI
- Project access control, quotas, audit records, and compute-node management
- CUDA, NPU, and ROCm model execution

## Common commands

```bash
sft-train env
sft-train bootstrap
sft-train api
sft-train webui
sft-train train --dry-run examples/text_lora_sft.yaml
```

Demo datasets are documented in [`data/`](data/README.md), configuration examples in [`examples/`](examples/README.md), and the CUDA container in [`docker/docker-cuda/`](docker/docker-cuda/README.md).

The versioned API contract is available at [`docs/openapi-v1.json`](docs/openapi-v1.json).
