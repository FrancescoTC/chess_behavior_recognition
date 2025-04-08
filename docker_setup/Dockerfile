FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="disi-Unibo-NLP"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface

# Set work directory
WORKDIR /workspace
ENV APP_PATH=/app

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y curl \
                       git \
                       bash \
                       nano \
                       python3.11 \
		       vim \
                       python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# COPY build/requirements.txt .
RUN pip3 install --no-cache-dir huggingface_hub datasets torch pandas bitsandbytes triton unsloth_zoo unsloth trl transformers vllm
RUN pip3 install --upgrade mistral_common
#ENTRYPOINT ["make", "-f", "/app/Makefile"]

# Copy login script
COPY login.sh /workspace/login.sh
RUN chmod +x /workspace/login.sh
ENTRYPOINT ["/workspace/login.sh"]
# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
