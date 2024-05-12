from gradio_client import Client

# This function interacts with the RKLLM model by calling the Gradio Client API
def chat_with_rkllm(user_message, history=[]):
    # Instantiate the Gradio Client, users need to modify this according to the specific URL where it is deployed
    client = Client("http://172.16.10.102:8080/")

    # Call the Gradio Client API for interaction; the internal APIs mainly include:
    # /get_user_input: The model retrieves user input and adds it to the history record
    # /get_RKLLM_output: RKLLM uses the history record containing the input to generate a response
    _, history = client.predict(user_message=user_message, history=history, api_name="/get_user_input")
    result_history = client.predict(history=history, api_name="/get_RKLLM_output")
    return result_history

if __name__ == '__main__':
    # Initialize chat history
    result_history = []

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
                # Call the chat_with_rkllm function to get the model's response
                result_history = chat_with_rkllm(user_message, result_history)

                # Print the model's output
                print("Q:", result_history[-1][0])
                print("A:", result_history[-1][1])
        except KeyboardInterrupt:
            print("\n")
            print("============================")
            print("Exiting the program......")
            print("============================")
            break

            
