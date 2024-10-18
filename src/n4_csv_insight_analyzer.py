from dotenv import load_dotenv 
from langchain_experimental.agents import create_pandas_dataframe_agent  # Create an agent to interact with Pandas DataFrames
from langchain_openai import OpenAI  
import pandas as pd  

load_dotenv()

# Function to handle CSV analysis based on user query
def csv_analyzer(csv_file_path, query):
     """
     This function processes a CSV file and uses a language model to respond to user queries 
     about the CSV data in a natural language manner.

     Args:
     csv_file_path (str): The path to the CSV file.
     query (str): The question or query in natural language regarding the CSV data.

     Returns:
     str: The response generated by the agent.
     """

     # Load the CSV file into a Pandas DataFrame
     df = pd.read_csv(csv_file_path)

     # Initialize the language model (OpenAI) to be used for generating responses
     llm = OpenAI()

     # Create an agent that allows interaction with the DataFrame using the language model
     agent = create_pandas_dataframe_agent(llm, df, verbose=True)

     # Generate a response to the user query by interacting with the agent
     return agent.invoke(query)


# Hardcoded user input for terminal execution
if __name__ == "__main__":
     csv_file_path = "../data/employees.csv" 

     user_query = "What is the average sales for 2023?"

     print(f"CSV File Path: {csv_file_path}")
     print(f"User Query: {user_query}")

     # Call the csv_analyzer function and get the response
     response = csv_analyzer(csv_file_path, user_query)

     print("Response to your query:")
     print(response)
