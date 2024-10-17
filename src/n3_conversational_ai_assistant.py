from langchain_openai import OpenAI  # LangChain's OpenAI wrapper
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationSummaryMemory  # Used to store and summarize conversation history
)

API_Key = "your_openai_api_key" 
user_input = "Tell me about the future of AI."  

# Function to initialize the conversation and get a response
def getGenResponse(userInput, api_key):
    """
    This function initializes the conversation chain and returns the model's response to the user's input.
    If the conversation does not exist, it creates a new one with the appropriate model and memory.
    """
    
    # Check if a conversation session has already been initialized.
    # If not, it sets up the LLM and memory for the conversation.
    if 'conversation' not in globals():
        global conversation  # Store conversation globally for reuse across multiple calls
        
        # Initialize the OpenAI LLM (GPT-3.5 Turbo, in this case)
        llm = OpenAI(
            temperature=0,  # Temperature controls randomness; 0 is deterministic
            openai_api_key=api_key,  # API Key for OpenAI
            model_name='gpt-3.5-turbo-instruct'  # Model name (default is GPT-3.5-turbo)
        )

        # Initialize the conversation chain with memory to summarize and store the conversation history
        conversation = ConversationChain(
            llm=llm,
            verbose=True,  # Enables detailed logging of the interaction
            memory=ConversationSummaryMemory(llm=llm)  # Uses memory to keep track of conversation context
        )

    # Get the model's response based on the user's input
    response = conversation.predict(input=userInput)
    
    # Print the conversation memory buffer, showing the history of interactions
    print(conversation.memory.buffer)

    return response  # Return the model's response

# Hardcoded process for interaction
# Normally, user input would be dynamic, but here we hardcode it for simplicity.
print("User: ", user_input)
model_response = getGenResponse(user_input, API_Key)  # Call the getGenResponse function with the user's input
print("AI: ", model_response)  # Output the AI's response
