
# Day 17 - Text Chunking

text = """
Generative AI is a technology that creates new content.
Large Language Models understand and generate human language.
RAG stands for Retrieval Augmented Generation.
RAG retrieves information from documents before generating answers.
Text chunking divides large documents into smaller pieces.
"""

# Split text into words
words = text.split()

# Set chunk size
chunk_size = 10

# Create chunks
chunks = []

for i in range(0, len(words), chunk_size):
    chunk = words[i:i + chunk_size]
    chunks.append(" ".join(chunk))

# Display chunks
print("Original Text:")
print(text)

print("\nText Chunks:")

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)
