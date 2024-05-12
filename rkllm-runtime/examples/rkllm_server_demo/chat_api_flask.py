import sys
import requests
import json

# Set the server's URL
server_url = 'http://172.16.10.102:8080/rkllm_chat'
# Set whether to enable streaming conversations
is_streaming = True

# Create a session object
session = requests.Session()
session.keep_alive = False  # Disable connection pooling to maintain a long connection
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount('https://', adapter)
session.mount('http://', adapter)

if __name__ == '__main__':
    print("============================")
    print("Enter your question in the terminal to chat with the RKLLM model....")
    print("============================")
    # Enter a loop to continuously get user input and chat with the RKLLM model
    while True:
        try:
            user_message = input("Please enter your question: ")
            if user_message == "exit":
                print("============================")
                print("Exiting the program......")
                print("============================")
                break
            else:
                # Set request headers, which have no actual effect here, only to mimic the OpenAI interface
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'not_required'
                }

                # Prepare the data to send
                # model: The model defined when setting up the RKLLM-Server, which has no effect here
                # messages: The user's input question, which the RKLLM-Server will take as input and return the model's response; supports adding multiple questions in messages
                # stream: Whether to enable streaming conversation, similar to the OpenAI interface
                data = {
                    "model": 'your_model_deploy_with_RKLLM_Server',
                    "messages": [{"role": "user", "content": user_message}],
                    "stream": is_streaming
                }

                # Send POST request
                responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)

                if not is_streaming:
                    # Parse the response
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", json.loads(responses.text)["choices"][-1]["message"]["content"])
                    else:
                        print("Error:", responses.text)
                else:
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", end="")
                        for line in responses.iter_lines():
                            if line:
                                line = json.loads(line.decode('utf-8'))
                                if line["choices"][-1]["finish_reason"] != "stop":
                                    print(line["choices"][-1]["delta"]["content"], end="")
                                    sys.stdout.flush()
                    else:
                        print('Error:', responses.text)



        except KeyboardInterrupt:
            # Catch Ctrl-C signal, close the session
            session.close()

            print("\n")
            print("============================")
            print("Exiting the program......")
            print("============================")
            break

