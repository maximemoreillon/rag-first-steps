import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Initialize SentenceTransformer for embedding the text
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for storing the embeddings
dimension = 384  # Dimensions of 'all-MiniLM-L6-v2' embeddings
index = faiss.IndexFlatL2(dimension)  # Flat index, you can use other FAISS indices for better performance

# Sample documents to add to the FAISS index
documents = [
    "Abbey's last name comes before Tonnie's alphabetically.",
    "The 14 year old doesn't have the names Hannah, Ann, or Marino.",
    "No one's first name starts with the same letter as her middle name.",
    "Hannah is the oldest and has the shortest last name.",
    "Kayla's last name starts with the same letter as her first.",
    "Tonnie's middle name has the same number of letters as her first.",
    "Kayla's and Abbey's middle names both start with vowels.",
    "Tonnie isn't the youngest.",
]

# Embed documents into vectors
document_embeddings = embedding_model.encode(documents)

# Add document embeddings to the FAISS index
index.add(np.array(document_embeddings))

# Function to retrieve the most relevant documents based on the query
def retrieve_relevant_documents(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)  # D: distances, I: indices
    relevant_docs = [documents[i] for i in I[0]]
    return relevant_docs

# Function to generate a response using OpenAI GPT-3 or GPT-4
def generate_response(query, relevant_docs):
    context = "\n".join(relevant_docs)

    prompt = f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Main RAG flow
query =  "Who is the youngest?"
    
# Step 1: Retrieve relevant documents from the FAISS index
relevant_docs = retrieve_relevant_documents(query)
print("Relevant Documents:")
for doc in relevant_docs:
    print(f"- {doc}")

# Step 2: Generate a response using OpenAI model
answer = generate_response(query, relevant_docs)
print("\nAnswer from the model:")
print(answer)