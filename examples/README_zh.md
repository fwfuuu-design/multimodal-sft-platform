# 配置示例

[English](README.md)

本目录提供平台常用任务的配置示例。

- `text_lora_sft.yaml`：文本 LoRA SFT
- `image_qlora_sft.yaml`：图文 4-bit bitsandbytes QLoRA SFT
- `adapter_merge.yaml`：将 Adapter 合并到未量化基础模型

配置中的模型和输出路径是占位路径，可根据运行环境进行调整。训练配置可通过以下命令验证：

```bash
sft-train train --dry-run examples/text_lora_sft.yaml
```
