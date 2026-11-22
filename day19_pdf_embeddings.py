
# Day 19 - PDF Text Embeddings

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Convert chunks into numerical vectors
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(chunks)

print("PDF TEXT EMBEDDINGS")

print("\nNumber of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst chunk embedding:")
print(embeddings.toarray()[0])

print("\nPDF Embedding Completed!")
