import google.generativeai as genai

# Configure the API key
genai.configure(api_key="your_api_key")  

def chat_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Main program
print("Welcome to Gemini Chat! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye")
        break
    response = chat_with_gemini(user_input)
    print(f"Gemini: {response}")
