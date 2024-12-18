# Import required libraries
from txtai.embeddings import Embeddings
import openai

# Initialize txtai embeddings
embeddings = Embeddings()

# Sample data to index
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
query = "Who is the youngest?"
response = query_txtai_with_openai(query)
print(response)