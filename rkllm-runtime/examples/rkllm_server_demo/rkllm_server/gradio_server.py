import ctypes
import os
import sys
import threading
import time
import json
import gradio as gr

# Paths
MODEL_PATH = "/models"
CONFIG_PATH = "/rkllm-runtime/examples/rkllm_api_demo/src/models_config.json"

# Global variables for callback
global_text = []
global_state = -1
split_byte_data = bytes(b"")

# Load model configurations from JSON
def load_model_config(model_name):
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as config_file:
        configs = json.load(config_file)
    return configs.get("models", {}).get(model_name, {})

# Get available models in the /models directory
def get_available_models():
    if not os.path.exists(MODEL_PATH) or not os.path.isdir(MODEL_PATH):
        return ["No models available"]
    return [f for f in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, f))]

# RKLLM wrapper class
class RKLLM:
    def __init__(self, model_path, config):
        self.rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')
        rkllm_param = RKLLMParam()

        # Set parameters from JSON config
        rkllm_param.model_path = bytes(model_path, 'utf-8')
        rkllm_param.max_context_len = config.get("max_context_len", 512)
        rkllm_param.max_new_tokens = config.get("max_new_tokens", 256)
        rkllm_param.temperature = config.get("temperature", 0.8)
        rkllm_param.top_k = config.get("top_k", 40)
        rkllm_param.top_p = config.get("top_p", 0.9)

        # Initialize model
        self.handle = ctypes.c_void_p()
        self.rkllm_lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), None)

    def run(self, prompt):
        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = 0
        rkllm_input.input_data.prompt_input = bytes(prompt, 'utf-8')
        self.rkllm_lib.rkllm_run(self.handle, ctypes.byref(rkllm_input), None, None)

    def release(self):
        self.rkllm_lib.rkllm_destroy(self.handle)

# Callback implementation
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == 2:  # RKLLM_RUN_FINISH
        global_state = state
    elif state == 3:  # RKLLM_RUN_ERROR
        global_state = state
        global_text.append("Error occurred during LLM inference.")
    else:
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text

# Callback type
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# Record user input
def get_user_input(user_message, history, selected_model):
    if not selected_model or selected_model == "No models available":
        history = history + [["Please select a model from the dropdown.", None]]
        return "", history

    config = load_model_config(selected_model)
    prompt_prefix = config.get("PROMPT_TEXT_PREFIX", "<|im_start|>system ")
    prompt_postfix = config.get("PROMPT_TEXT_POSTFIX", "<|im_end|><|im_start|>assistant")
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    prompt = f"{prompt_prefix}{system_prompt}{prompt_postfix} {user_message} {prompt_postfix}"
    history = history + [[f"Model: {selected_model}\nUser: {user_message}", None]]
    return prompt, history

# Stream model output
def get_RKLLM_output(history, prompt, selected_model):
    global global_text, global_state
    global_text = []
    global_state = -1

    model_path = os.path.join(MODEL_PATH, selected_model)
    config = load_model_config(selected_model)
    rkllm_model = RKLLM(model_path, config)

    model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
    model_thread.start()

    history[-1][1] = ""
    model_thread_finished = False
    while not model_thread_finished:
        while len(global_text) > 0:
            history[-1][1] += global_text.pop(0)
            time.sleep(0.005)
            yield history

        model_thread.join(timeout=0.005)
        model_thread_finished = not model_thread.is_alive()

    rkllm_model.release()

# Create Gradio interface
with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
    gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
    gr.Markdown("### Enter your question in the inputTextBox and press the Enter key to chat with the RKLLM model.")

    available_models = get_available_models()
    model_dropdown = gr.Dropdown(choices=available_models, label="Select Model", value=available_models[0] if available_models else "No models available")
    rkllmServer = gr.Chatbot(height=600)
    msg = gr.Textbox(placeholder="Please input your question here...", label="Input")
    clear = gr.Button("Clear")

    msg.submit(get_user_input, [msg, rkllmServer, model_dropdown], [msg, rkllmServer]).then(
        get_RKLLM_output, [rkllmServer, msg, model_dropdown], rkllmServer
    )
    clear.click(lambda: None, None, rkllmServer, queue=False)

    chatRKLLM.queue()
    chatRKLLM.launch()

# Ensure graceful shutdown of resources
try:
    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
except Exception as e:
    print(f"Error releasing RKLLM model resources: {e}")


