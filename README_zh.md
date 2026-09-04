# 多模态微调平台

[English](README.md)

多模态微调平台提供文本与图文数据的监督微调工作流。

## 主要功能

- LoRA 与 4-bit bitsandbytes QLoRA 训练
- 评估、预测、图文对话与 Adapter 导出
- 带身份认证的 API 和 WebUI
- 项目权限、配额、审计与计算节点管理
- CUDA、NPU 和 ROCm 模型运行环境

## 常用命令

```bash
sft-train env
sft-train bootstrap
sft-train api
sft-train webui
sft-train train --dry-run examples/text_lora_sft.yaml
```

示例数据见 [`data/`](data/README_zh.md)，配置示例见 [`examples/`](examples/README_zh.md)，CUDA 容器说明见 [`docker/docker-cuda/`](docker/docker-cuda/README.md)。

版本化 API 契约位于 [`docs/openapi-v1.json`](docs/openapi-v1.json)。
