
# Day 19 - Semantic Search

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read PDF
pdf = PdfReader("documents/sample.pdf")

text = ""

for page in pdf.pages:
    text += page.extract_text() or ""

# Split text into chunks
words = text.split()
chunk_size = 25

chunks = []

for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size])
    chunks.append(chunk)

# Convert chunks into TF-IDF vectors
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

# Ask a question
query = input("Ask a question about your PDF: ")

# Convert question into a vector
query_vector = vectorizer.transform([query])

# Calculate similarity
similarities = cosine_similarity(query_vector, chunk_vectors)

# Find the most relevant chunk
best_index = similarities.argmax()

print("\nMost Relevant Information:")
print(chunks[best_index])

print("\nSimilarity Score:", round(similarities[0][best_index], 3))

print("\nSemantic Search Completed!")
