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
git clone https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
git clone https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct

# Done here to avoid cloning full repository for the Docker image
apt install -y git-lfs

# Needed
DEBIAN_FRONTEND=noninteractive apt install -y python3-tk

# cd Qwen-1_8B-Chat
# git lfs pull
