#!/bin/bash

# Always useful
apt update && apt full-upgrade -y

# Ensure models directory exists
if [ ! -d /models ]; then
    mkdir -p /models
    chmod -R 777 /models  # Ensure permissions
    echo "Created /models directory with appropriate permissions."
fi

# Update repo without updating image
cd /ezrknn-llm
git pull

# Start a bash session for interaction
bash

