import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# Create a lock to control server access when multiple users are connected
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocking state
is_blocking = False

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

# Define global variables to save callback function outputs, useful for displaying in the Gradio interface
global_text = []
global_state = -1
split_byte_data = bytes(b"")  # Used to store split byte data

# Define a structure in the dynamic library
class Token(ctypes.Structure):
    _fields_ = [
        ("logprob", ctypes.c_float),
        ("id", ctypes.c_int32)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("tokens", ctypes.POINTER(Token)),
        ("num", ctypes.c_int32)
    ]

# Define a callback function
def callback(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == 0:
        # Save the token text output and RKLLM running state
        global_state = state
        # Monitor if the current byte data is complete; if not, record it for later parsing
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            print((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()
    elif state == 1:
        # Save the RKLLM running state
        global_state = state
        print("\n")
        sys.stdout.flush()
    else:
        print("run error")

# Connect the Python-side to the C++-side callback function
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
c_callback = callback_type(callback)

# Define a structure in the dynamic library
class RKNNllmParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("num_npu_core", ctypes.c_int32),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("logprobs", ctypes.c_bool),
        ("top_logprobs", ctypes.c_int32),
        ("use_gpu", ctypes.c_bool)
    ]

# Define RKLLM_Handle_t and userdata
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

# Set prompt text
PROMPT_TEXT_PREFIX = "system You are a helpful assistant.  user"
PROMPT_TEXT_POSTFIX = "assistant"

# Define the RKLLM class on the Python side, including initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM:
    def __init__(self, model_path, target_platform):
        rknnllm_param = RKNNllmParam()
        rknnllm_param.model_path = bytes(model_path, 'utf-8')
        if target_platform == "rk3588":
            rknnllm_param.num_npu_core = 3
        elif target_platform == "rk3576":
            rknnllm_param.num_npu_core = 1
        rknnllm_param.max_context_len = 320
        rknnllm_param.max_new_tokens = 512
        rknnllm_param.top_k = 1
        rknnllm_param.top_p = 0.9
        rknnllm_param.temperature = 0.8
        rknnllm_param.repeat_penalty = 1.1
        rknnllm_param.frequency_penalty = 0.0
        rknnllm_param.presence_penalty = 0.0
        rknnllm_param.mirostat = 0
        rknnllm_param.mirostat_tau = 5.0
        rknnllm_param.mirostat_eta = 0.1
        rknnllm_param.logprobs = False
        rknnllm_param.top_logprobs = 5
        rknnllm_param.use_gpu = True
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKNNllmParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), rknnllm_param, c_callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

    def run(self, prompt):
        prompt = bytes(PROMPT_TEXT_PREFIX + prompt + PROMPT_TEXT_POSTFIX, 'utf-8')
        self.rkllm_run(self.handle, prompt, ctypes.byref(userdata))
        return

    def release(self):
        self.rkllm_destroy(self.handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_platform', help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--rkllm_model_path', help='Absolute path of the converted rkllm model on the Linux board')
    args = parser.parse_args()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("====== Error: Please specify the correct target platform: rk3588/rk3576 ======")
        sys.stdout.flush()
        exit()

    if not os.path.exists(args.rkllm_model_path):
        print("====== Error: Please provide the accurate path to the rkllm model, ensuring it is the absolute path on the board ======")
        sys.stdout.flush()
        exit()

    # Frequency setting
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set file descriptor limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize the RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    target_platform = args.target_platform
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, target_platform)
    print("RKLLM initialization successful!")
    print("==============================")
    sys.stdout.flush()

    # Create a function to receive user data sent via request
    @app.route('/rkllm_chat', methods=['POST'])
    def receive_message():
        # Link global variables, get output information from the callback function
        global global_text, global_state
        global is_blocking

        # If the server is in a blocking state, return a specific response
        if is_blocking or global_state==0:
            return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
        
        # Acquire the lock
        lock.acquire()
        try:
            # Set the server to a blocking state
            is_blocking = True

            # Get JSON data from the POST request
            data = request.json
            if data and 'messages' in data:
                # Reset global variables
                global_text = []
                global_state = -1

                # Define the return structure
                rkllm_responses = {
                    "id": "rkllm_chat",
                    "object": "rkllm_chat",
                    "created": None,
                    "choices": [],
                    "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                    }
                }

                if not "stream" in data.keys() or data["stream"] == False:
                    # Handle received data here
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        # Create a thread for model inference
                        model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                        model_thread.start()

                        # Wait for the model to complete, periodically check the inference thread
                        model_thread_finished = False
                        while not model_thread_finished:
                            while len(global_text) > 0:
                                rkllm_output += global_text.pop(0)
                                time.sleep(0.005)

                            model_thread.join(timeout=0.005)
                            model_thread_finished = not model_thread.is_alive()
                        
                        rkllm_responses["choices"].append(
                            {"index": index,
                            "message": {
                                "role": "assistant",
                                "content": rkllm_output,
                            },
                            "logprobs": None,
                            "finish_reason": "stop"
                            }
                        )
                    return jsonify(rkllm_responses), 200
                else:
                    # Handle received data here
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        def generate():
                            # Create a thread for model inference
                            model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                            model_thread.start()

                            # Wait for the model to complete, periodically check the inference thread
                            model_thread_finished = False
                            while not model_thread_finished:
                                while len(global_text) > 0:
                                    rkllm_output = global_text.pop(0)

                                    rkllm_responses["choices"].append(
                                        {"index": index,
                                        "delta": {
                                            "role": "assistant",
                                            "content": rkllm_output,
                                        },
                                        "logprobs": None,
                                        "finish_reason": "stop" if global_state == 1 else None,
                                        }
                                    )
                                    yield f"{json.dumps(rkllm_responses)}\n\n"

                                model_thread.join(timeout=0.005)
                                model_thread_finished = not model_thread.is_alive()

                    return Response(generate(), content_type='text/plain')
            else:
                return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
        finally:
            # Release the lock
            lock.release()
            # Set the server state to non-blocking
            is_blocking = False
        
    # Start the Flask application
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("RKLLM model inference complete, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")

