
# Day 18 - Vector Database Practical

import chromadb

# Create an in-memory vector database
client = chromadb.Client()

# Create a collection
collection = client.create_collection(
    name="my_documents"
)

# Store documents and their embeddings
collection.add(
    ids=["1", "2", "3"],
    documents=[
        "Python is a programming language",
        "Machine learning helps computers learn",
        "RAG retrieves information from documents"
    ],
    embeddings=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
)

# Retrieve similar documents
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=2
)

print("Retrieved Documents:")

for document in results["documents"][0]:
    print(document)
