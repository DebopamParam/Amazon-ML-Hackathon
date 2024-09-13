from langchain_ollama import OllamaLLM

# Initialize the model
model = OllamaLLM(model="llama3.1:8b")

# Function to invoke the model
def invoke_model(prompt):
    result = model.invoke(prompt)
    print(result)

# Keep the script running
try:
    while True:
        prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        invoke_model(prompt)
except KeyboardInterrupt:
    print("\nScript terminated by user")