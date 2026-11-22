
# Day 20 - Chat With Your Documents (RAG Project)

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("===================================")
print("       CHAT WITH YOUR PDF")
print("===================================")

# Step 1: Read PDF
pdf = PdfReader("documents/sample.pdf")

text = ""

for page in pdf.pages:
    text += page.extract_text() or ""

# Step 2: Split text into chunks
words = text.split()
chunk_size = 40

chunks = []

for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size])
    chunks.append(chunk)

# Step 3: Generate embeddings
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

print("\nPDF loaded successfully!")
print("Total chunks:", len(chunks))

# Step 4: Chat with the PDF
while True:

    question = input("\nAsk a question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Thank you for using the RAG application!")
        break

    # Step 5: Retrieve relevant information
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector, chunk_vectors
    )

    best_index = similarities.argmax()
    score = similarities[0][best_index]

    # Step 6: Display retrieved context
    if score == 0:
        print("\nNo relevant information found in the PDF.")
    else:
        print("\nRelevant information from your PDF:")
        print(chunks[best_index])

        print("\nSimilarity score:", round(score, 3))
