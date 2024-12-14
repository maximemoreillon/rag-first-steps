from txtai import Embeddings
import openai

# Create embeddings instance with a semantic graph
embeddings = Embeddings({
  "autoid": "uuid5",
  "path": "intfloat/e5-base",
  "instructions": {
    "query": "query: ",
    "data": "passage: "
  },
  "content": True,
  "graph": {
      "approximate": False,
      "topics": {}
  }
})

# Load dataset
wikipedia = Embeddings()
wikipedia.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

# LIMIT originally 100000 but reduced so as to run on my laptop
query = """
SELECT id, text FROM txtai
order by percentile desc
LIMIT 1000
"""

embeddings.index(wikipedia.search(query))


g = embeddings.graph.search("""
MATCH P=({id: "Roman Empire"})-[*1..3]->({id: "Saxons"})-[*1..3]->({id: "Vikings"})-[*1..3]->({id: "Battle of Hastings"})
RETURN P
LIMIT 20
""", graph=True)


context = "\n".join(g.attribute(node, "text") for node in list(g.scan()))

prompt = "Who were the enemies of the Roman Empire?"

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Based on the following context, answer the query: {prompt}\n\nContext:\n{context}"}
    ]
)

content = response.choices[0].message.content

print(content)
