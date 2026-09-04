# https://hub.docker.com/r/pytorch/pytorch/tags
ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"

# Copy the application into the image
COPY . /app

# Install Multimodal SFT Platform
RUN pip install --no-cache-dir --no-build-isolation -e . && \
    pip install --no-cache-dir --no-build-isolation -r requirements/metrics.txt -r requirements/bitsandbytes.txt

# Rebuild flash attention
RUN if [ "${INSTALL_FLASHATTN}" == "true" ]; then \
        pip uninstall -y ninja && \
        pip install --no-cache-dir ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# Set up volumes
# VOLUME [ "/root/.cache/huggingface", "/app/shared_data", "/app/output" ]

# Expose the product WebUI port
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Expose port 8000 for API service
ENV API_PORT=8000
EXPOSE 8000

# unset proxy
ENV http_proxy=
ENV https_proxy=

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
