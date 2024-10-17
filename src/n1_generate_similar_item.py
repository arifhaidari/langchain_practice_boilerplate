import os  
from langchain_openai import OpenAIEmbeddings  # To generate embeddings for similarity search
from langchain_community.vectorstores import FAISS  # For similarity search using FAISS
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAIEmbeddings object to generate vector embeddings
embeddings = OpenAIEmbeddings()

# Import CSVLoader to load data from a CSV file
from langchain.document_loaders.csv_loader import CSVLoader

# File path to the CSV, located in the 'data' folder outside the 'src' folder
csv_file_path = '../data/myData.csv'

# Load data from the CSV file
loader = CSVLoader(file_path=csv_file_path, csv_args={
    'delimiter': ',',  # CSV field delimiter
    'quotechar': '"',  # Character to denote quoted fields
    'fieldnames': ['Words']  # The fieldnames for the CSV columns
})

# Load the data from the CSV file into the 'data' variable
data = loader.load()

# Display the loaded data in the terminal
print("Loaded data from CSV:")
print(data)

# Initialize the FAISS vector store with the documents and embeddings
db = FAISS.from_documents(data, embeddings)

# user input for similarity search
user_input = "apple"  

# Perform similarity search using the hardcoded input
docs = db.similarity_search(user_input)

# Display the top matches
print("\nTop Matches:")
print(docs[0])  # Print the first match
print(docs[1].page_content)  # Print the content of the second match

"""
Faiss is a library — developed by Facebook AI — that enables efficient similarity search. 
So, given a set of vectors, we can index them using Faiss — then using another 
vector (the query vector), we search for the most similar vectors within the index.

------

FAISS: is more suitable for those who need a customizable, efficient local solution for 
vector search and are comfortable managing their own infrastructure.
Pinecone: is ideal for companies and teams that need large-scale, hassle-free vector 
search with real-time updates, without dealing with infrastructure or scalability challenges.
"""