// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Modified by Pelochus

#include <string>
#include <iostream>
#include <fstream>
#include <csignal>
#include <vector>
#include "rkllm.h"
#include "json.hpp" // nlohmann/json header

using namespace std;
using json = nlohmann::json;

LLMHandle llmHandle = nullptr;

// Exit handler
void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        cout << "Caught exit signal. Exiting..." << endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
        exit(signal);
    }
}

// Callback function
void callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == LLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == LLM_RUN_ERROR)
    {
        printf("\\LLM run error\n");
    }
    else
    {
        printf("%s", result->text);
    }
}

// Load model configuration from JSON file
bool load_model_config(const string &config_path, const string &model_name, json &model_config)
{
    ifstream config_file(config_path);
    if (!config_file.is_open())
    {
        cerr << "Error: Unable to open configuration file: " << config_path << endl;
        return false;
    }

    json config;
    config_file >> config;
    config_file.close();

    if (config.contains("models") && config["models"].contains(model_name))
    {
        model_config = config["models"][model_name];
        string family = model_config.value("family", "");

        if (!family.empty() && config.contains("families") && config["families"].contains(family))
        {
            // Merge family-level defaults into model config
            for (auto &el : config["families"][family].items())
            {
                if (!model_config.contains(el.key()))
                {
                    model_config[el.key()] = el.value();
                }
            }
        }
        return true;
    }
    else
    {
        cerr << "Error: Model configuration for " << model_name << " not found in the configuration file." << endl;
        return false;
    }
}

string interpolate_prompt(const string &prefix, const string &system_prompt)
{
    // Replace placeholder {{system_prompt}} in the prefix
    size_t pos = prefix.find("{{system_prompt}}");
    if (pos != string::npos)
    {
        return prefix.substr(0, pos) + system_prompt + prefix.substr(pos + 16); // 16 is the length of "{{system_prompt}}"
    }
    return prefix;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s [rkllm_model_path] [model_name]\n", argv[0]);
        return -1;
    }

    signal(SIGINT, exit_handler);

    string rkllm_model(argv[1]);
    string model_name(argv[2]);

    printf("RKLLM starting, please wait...\n");

    // Load model-specific configuration
    json model_config;
    if (!load_model_config("/path/to/models_config.json", model_name, model_config))
    {
        return -1;
    }

    // Set model parameters
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = rkllm_model.c_str();
    param.num_npu_core = 2;
    param.max_new_tokens = model_config.value("max_new_tokens", 256);
    param.max_context_len = model_config.value("max_context_len", 512);
    param.top_k = 1;
    param.logprobs = false;
    param.top_logprobs = 5;
    param.use_gpu = false;

    // Initialize RKLLM
    if (rkllm_init(&llmHandle, &param, callback) != 0)
    {
        cerr << "Error: Failed to initialize RKLLM model." << endl;
        return -1;
    }
    printf("RKLLM init success!\n");

    // Extract dynamic prompt settings
    string prompt_prefix = model_config.value("PROMPT_TEXT_PREFIX", "");
    string system_prompt = model_config.value("system_prompt", "");
    string prompt_postfix = model_config.value("PROMPT_TEXT_POSTFIX", "");
    string interpolated_prefix = interpolate_prompt(prompt_prefix, system_prompt);

    // Display welcome message
    vector<string> pre_input;
    pre_input.push_back("Welcome to ezrkllm! This is an adaptation of Rockchip's rknn-llm repo.");
    pre_input.push_back("To exit the model, enter either 'exit' or 'quit'.");
    pre_input.push_back("More information here: https://github.com/Pelochus/ezrknpu");

    cout << "\n*************************** Pelochus' ezrkllm runtime *************************\n" << endl;
    for (const auto &line : pre_input)
    {
        cout << line << endl;
    }
    cout << "\n*******************************************************************************\n" << endl;

    // Main loop
    string text;
    while (true)
    {
        string input_str;
        printf("\nYou: ");
        getline(cin, input_str);

        if (input_str == "exit" || input_str == "quit")
        {
            cout << "Quitting program..." << endl;
            break;
        }

        string query = interpolated_prefix + input_str + prompt_postfix;

        printf("LLM: ");
        rkllm_run(llmHandle, query.c_str(), nullptr);
    }

    rkllm_destroy(llmHandle);

    return 0;
}


