import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import gradio as gr
import argparse

# Set environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "8080"

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

# Define global variables to save callback function outputs for display in the gradio interface
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

# Define a structure in the dynamic library for RKLLM model parameters
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
            rknnllm_param.num_npu_core = 2
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

    # Record user input prompt         
    def get_user_input(user_message, history):
        history = history + [[user_message, None]]
        return "", history

    # Get RKLLM model output and perform streaming printing
    def get_RKLLM_output(history):
        # Link global variables, get output information from the callback function
        global global_text, global_state
        global_text = []
        global_state = -1

        # Create a thread for model inference
        model_thread = threading.Thread(target=rkllm_model.run, args=(history[-1][0],))
        model_thread.start()

        # history[-1][1] represents the current output conversation
        history[-1][1] = ""
        
        # Wait for the model to complete, periodically check the inference thread
        model_thread_finished = False
        while not model_thread_finished:
            while len(global_text) > 0:
                history[-1][1] += global_text.pop(0)
                time.sleep(0.005)
                # Gradio automatically pushes yield returned results for output when the then method is called
                yield history

            model_thread.join(timeout=0.005)
            model_thread_finished = not model_thread.is_alive()

    # Create a gradio interface
    with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
        gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
        gr.Markdown("### Type your question in the inputTextBox, press Enter, and chat with the RKLLM model.")
        # Create a Chatbot component to display conversation history
        rkllmServer = gr.Chatbot(height=600)
        # Create a Textbox component for user input
        msg = gr.Textbox(placeholder="Please input your question here...", label="inputTextBox")
        # Create a Button component to clear chat history
        clear = gr.Button("Clear")

        # Submit user input to the get_user_input function and immediately update the chat history
        # Then call the get_RKLLM_output function to further update the chat history
        # The queue=False parameter ensures that these updates are executed immediately, not queued
        msg.submit(get_user_input, [msg, rkllmServer], [msg, rkllmServer], queue=False).then(get_RKLLM_output, rkllmServer, rkllmServer)
        # When the clear button is clicked, perform a no-op (lambda: None) and immediately clear the chat history
        clear.click(lambda: None, None, rkllmServer, queue=False)

    # Enable the event queue system
    chatRKLLM.queue()
    # Launch the Gradio application
    chatRKLLM.launch()

    print("====================")
    print("RKLLM model inference complete, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
