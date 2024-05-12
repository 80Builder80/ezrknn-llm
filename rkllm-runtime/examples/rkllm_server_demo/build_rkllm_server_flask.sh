#!/bin/bash

#*****************************************************************************************#
# This script is a one-click setup script for the RKLLM-Server-Flask service
# Users can run this script to automatically deploy the RKLLM-Server-Flask service on a Linux board.
# Usage: ./build_rkllm_server_flask.sh [target platform: rk3588/rk3576] [RKLLM-Server working path] [absolute path of the converted rkllm model on the board]
# example: ./build_rkllm_server_flask.sh rk3588 /user/data/rkllm_server /user/data/rkllm_server/model.rkllm
#*****************************************************************************************#

#################### Check if pip/gradio libraries are already installed on the board ####################
# 1. Prepare the gradio environment on the board
adb shell << EOF

# Check if pip3 is installed
if ! command -v pip3 &> /dev/null; then
    echo "-------- pip3 not installed, installing now... --------"
    # Install pip3
    sudo apt update
    sudo apt install python3-pip -y
else
    echo "-------- pip3 is already installed --------"
fi

# Check if flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "-------- flask not installed, installing now... --------"
    # Install flask
    pip install flask==2.2.2 Werkzeug==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo "-------- flask is already installed --------"
fi

exit

EOF

#################### Push server-related files to the board ####################
# 2. Check if the path to push into the board exists
adb shell ls $2 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    # If the path does not exist, create the path
    adb shell mkdir -p $2
    echo "-------- rkllm_server working directory does not exist, directory created --------"
else
    echo "-------- rkllm_server working directory already exists --------"
fi

# 3. Update the librkllmrt.so file in ./rkllm_server/lib
cp ../../runtime/Linux/librkllm_api/aarch64/librkllmrt.so  ./rkllm_server/lib/

# 4. Push files to the Linux board
adb push ./rkllm_server $2

#################### Enter the board and start the server service ####################
# 5. Enter the board to start the server service
adb shell << EOF

cd $2/rkllm_server/
python3 flask_server.py --target_platform $1 --rkllm_model_path $3

EOF

