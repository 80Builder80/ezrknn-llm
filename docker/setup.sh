#!/bin/bash

# Dependencies
apt update
apt install -y python3 pip git curl wget nano sudo apt-utils

# Better than using the default from Rockchip
git clone https://github.com/80Builder80/ezrknn-llm/
curl https://raw.githubusercontent.com/80Builder80/ezrknn-llm/master/install.sh | sudo bash

# For running the test.py
pip install /ezrknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl

# Clone some compatible LLMs
cd /ezrknn-llm/rkllm-toolkit/examples/huggingface
git clone https://huggingface.co/c01zaut/Llama-3.1-8B-Instruct-rk3588-1.1.1
git clone https://huggingface.co/c01zaut/deepseek-coder-7b-instruct-v1.5-rk3588-1.1.1

# Done here to avoid cloning full repository for the Docker image
apt install -y git-lfs

# Needed
DEBIAN_FRONTEND=noninteractive apt install -y python3-tk

# cd Qwen-1_8B-Chat
# git lfs pull
