# CUDA container

This directory provides the CUDA container configuration for Multimodal SFT Platform.

- `Dockerfile.base`: CUDA and Python base image
- `Dockerfile`: platform application image
- `docker-compose.yml`: local container and NVIDIA GPU settings

From the `SFTPlatform` directory, build and open the container with:

```bash
docker compose -f docker/docker-cuda/docker-compose.yml build
docker compose -f docker/docker-cuda/docker-compose.yml run --rm --service-ports sft_platform
```

Runtime settings such as identity, storage, network, and credentials are supplied through the deployment environment.
