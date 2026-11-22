
# Day 18 - PDF Text Chunking

from pypdf import PdfReader

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

# Display chunks
print("PDF TEXT CHUNKS")

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)

print("\nTotal Chunks:", len(chunks))
