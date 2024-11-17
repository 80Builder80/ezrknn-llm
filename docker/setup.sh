#!/bin/bash

# Dependencies
apt update
apt install -y python3 pip git curl wget nano sudo apt-utils

# Clone the repository
git clone https://github.com/80Builder80/ezrknn-llm/

# Install ezrknn-llm
curl https://raw.githubusercontent.com/80Builder80/ezrknn-llm/master/install.sh | sudo bash

# Install required Python packages
pip install /ezrknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl

# Ensure Python dependencies for Gradio server are installed
pip install gradio>=4.24.0

# Ensure required directories
if [ ! -d /models ]; then
    mkdir -p /models
    chmod -R 777 /models
fi

