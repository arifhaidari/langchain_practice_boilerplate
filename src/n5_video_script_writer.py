# Required Imports
from langchain_openai import ChatOpenAI  # OpenAI's LLM for generating content
from langchain.prompts import PromptTemplate  # Templates for structuring LLM prompts
from langchain.chains import LLMChain  # Create chains for sequential processing
from langchain_community.tools import DuckDuckGoSearchRun  # Use DuckDuckGo for web search

# Function to generate the YouTube video script
def script_generator(prompt, video_length, creativity, api_key):
     """
     Generates a YouTube or Twitter video title and script based on the user's prompt using OpenAI and DuckDuckGo search.

     Args:
     prompt (str): The topic or idea for the video.
     video_length (str): Expected length of the video in minutes.
     creativity (float): Creativity level (0.0 for low, 1.0 for high).
     api_key (str): OpenAI API key for accessing the LLM.

     Returns:
     tuple: Returns the search result, video title, and script.
     """

     # Template for generating the video title
     title_template = PromptTemplate(
          input_variables=['subject'],
          template="Please come up with a title for a YouTube video on the topic: {subject}."
     )

     # Template for generating the video script based on the title and search results
     script_template = PromptTemplate(
          input_variables=['title', 'DuckDuckGo_Search', 'duration'],
          template="Create a script for a YouTube video titled: '{title}' with a duration of {duration} minutes, "
                    "using the following search data: {DuckDuckGo_Search}."
     )

     # Initialize the OpenAI language model with the user's specified creativity level and API key
     llm = ChatOpenAI(temperature=creativity, openai_api_key=api_key, model_name='gpt-3.5-turbo')

     # Create chains for generating the video title and the script
     title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
     script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

     # Use DuckDuckGo search to gather information for script generation
     search = DuckDuckGoSearchRun()

     # Generate the video title based on the prompt
     title = title_chain.invoke(prompt)

     # Conduct a DuckDuckGo search based on the prompt
     search_result = search.run(prompt)

     # Generate the video script using the title, search results, and video duration
     script = script_chain.run(title=title, DuckDuckGo_Search=search_result, duration=video_length)

     # Return the search results, generated title, and the script
     return search_result, title, script


# Hardcoded user input for terminal execution
if __name__ == "__main__":
     api_key = "your_openai_api_key_here" 

     user_prompt = "How to improve productivity with AI tools"
     
     expected_video_length = "10"  # Example: 10 minutes

     # creativity level (0.0 = low creativity, 1.0 = high creativity)
     creativity_level = 0.7

     # Display the hardcoded inputs
     print(f"User Prompt: {user_prompt}")
     print(f"Expected Video Length: {expected_video_length} minutes")
     print(f"Creativity Level: {creativity_level}")

     # Generate the YouTube script based on the hardcoded inputs
     search_result, title, script = script_generator(user_prompt, expected_video_length, creativity_level, api_key)

     # Display the results in the terminal
     print("\nGenerated YouTube Video Title:")
     print(title)

     print("\nGenerated YouTube Video Script:")
     print(script)

     print("\nDuckDuckGo Search Results:")
     print(search_result)
