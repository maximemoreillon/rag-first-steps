# Import required libraries
from txtai.embeddings import Embeddings
import openai
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv

load_dotenv()

# Initialize txtai embeddings
embeddings = Embeddings()



articles = WikipediaLoader(query="Ethernet", load_max_docs=1, doc_content_chars_max=100000).load()
text = articles[0].page_content

# This chunking method is really simple and should be improved
chunk_size = 1024
documents = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Index documents into txtai
embeddings.index([(i, doc, None) for i, doc in enumerate(documents)])

# Function to query txtai and use OpenAI for insights
def query_txtai_with_openai(query):
    # Retrieve top 3 relevant results from txtai
    results = embeddings.search(query, 3)  # Results are document IDs
    
    # Combine retrieved results into a single context
    context = "\n".join([documents[doc_id] for doc_id, _ in results])

    # Call OpenAI's GPT model for generating insights
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the following context, answer the query: {query}\n\nContext:\n{context}"}
        ]
    )
    
    return response.choices[0].message.content

# Example usage
query = "Who invented Ethernet?"
response = query_txtai_with_openai(query)
print(response)