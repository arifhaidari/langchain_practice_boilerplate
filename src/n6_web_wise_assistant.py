import os
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain.document_loaders.sitemap import SitemapLoader

# these variables are supposed to be in .env (here only for testing and learning purpose)
HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"  # Replace with your actual HuggingFace API key
PINECONE_API_KEY = "your_pinecone_api_key_here"  # Replace with your actual Pinecone API key
PINECONE_ENVIRONMENT = "pinecone_environment_goes_here" # in web of pinecone
PINECONE_INDEX = "web_wise_assistant_index" # in web portal of pinecone
WEBSITE_URL = "website_url_goes_here"

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Function to fetch data from the website using the SitemapLoader
def scrape_url(sitemap_url):
     """
     Fetches the website data by scraping the sitemap.

     Args:
     sitemap_url (str): The URL of the sitemap to scrape.

     Returns:
     List[Document]: List of documents containing the website data.
     """
     loop = asyncio.new_event_loop()
     asyncio.set_event_loop(loop)
     
     # Load the website data from the provided sitemap URL
     loader = SitemapLoader(sitemap_url)
     docs = loader.load()
     return docs

# Function to split the scraped website data into chunks for easier processing
def data_splitter(docs):
     """
     Splits the website data into smaller chunks for processing.

     Args:
     docs (List[Document]): List of website data documents.

     Returns:
     List[Document]: List of smaller document chunks.
     """
     # Using RecursiveCharacterTextSplitter to split the data into chunks
     text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000,  # Size of each chunk
          chunk_overlap=200,  # Overlap between chunks for context
          length_function=len  # Define the function to calculate length
     )
     docs_chunks = text_splitter.split_documents(docs)
     return docs_chunks

# Function to push the processed data to Pinecone for creating a vector store
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
     """
     Pushes the website data to Pinecone by creating a vector store.

     Args:
     pinecone_apikey (str): API key for Pinecone.
     pinecone_environment (str): Environment for Pinecone (e.g., gcp-starter).
     pinecone_index_name (str): Name of the Pinecone index.
     embeddings (Embeddings): Embeddings instance to convert text into vectors.
     docs (List[Document]): List of documents to push to Pinecone.
     
     Returns:
     Index: The created Pinecone index.
     """
     # Initialize the Pinecone client with the provided API key and environment
     PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)

     # Create the Pinecone index with the given documents and embeddings
     index = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
     return index

# Function to pull the existing index data from Pinecone
def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
     """
     Pulls the existing index data from Pinecone for similarity search.

     Args:
     pinecone_apikey (str): API key for Pinecone.
     pinecone_environment (str): Environment for Pinecone (e.g., gcp-starter).
     pinecone_index_name (str): Name of the Pinecone index.
     embeddings (Embeddings): Embeddings instance for text vectorization.

     Returns:
     Index: The Pinecone index for similarity search.
     """
     # Initialize the Pinecone client and pull the existing index
     PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)
     index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
     return index

# Function to perform a similarity search on the Pinecone index
def get_similar_docs(index, query, k=2):
     """
     Fetches the most relevant documents from Pinecone using similarity search.

     Args:
     index (Index): The Pinecone index to search.
     query (str): The user's query or prompt for similarity search.
     k (int): Number of top relevant documents to return.

     Returns:
     List[Document]: List of the most relevant documents.
     """
     # Perform similarity search on the index based on the user's query
     similar_docs = index.similarity_search(query, k=k)
     return similar_docs

# Main function to run the entire process
if __name__ == "__main__":
     # prompt and document count (user input simulation)
     prompt = "How does AI improve job recruitment processes?"  
     document_count = 3  # Number of top documents to retrieve

     print(f"User Prompt: {prompt}")
     print(f"Number of Documents to Retrieve: {document_count}")

     # Step 1: Fetch website data by scraping the given sitemap URL
     site_data = scrape_url(WEBSITE_URL)
     print("Website data retrieval done...")

     # Step 2: Split the website data into smaller chunks
     chunks_data = data_splitter(site_data)
     print("Data splitting done...")

     # Step 3: Create an embeddings instance for vector representation of text
     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
     print("Embeddings instance creation done...")

     # Step 4: Push the document chunks to Pinecone for indexing
     push_to_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX, embeddings, chunks_data)
     print("Data pushed to Pinecone successfully...")

     # Step 5: Retrieve the existing Pinecone index
     index = pull_from_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX, embeddings)
     print("Pinecone index retrieval done...")

     # Step 6: Perform similarity search on the Pinecone index with the user's prompt
     relevant_docs = get_similar_docs(index, prompt, document_count)
     print("Relevant documents retrieved...")

     # Step 7: Display the search results (relevant documents)
     for i, doc in enumerate(relevant_docs, 1):
          print(f"\nResult {i}:")
          print(f"Document Content: {doc.page_content}")
          print(f"Source Link: {doc.metadata['source']}")
