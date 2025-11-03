import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# Initialize client
client = OpenAI()

# Input text
text_input = "Artificial intelligence is transforming the world."

# Create embedding
response = client.embeddings.create(
    model="text-embedding-3-small",   # or "text-embedding-3-large"
    input=[text_input, text_input]
)

print(response)
# Extract embedding vector
embedding_vector = response.data[1].embedding

print(f"Embedding length: {len(embedding_vector)}")
print(embedding_vector[:10])  # show first 10 numbers
