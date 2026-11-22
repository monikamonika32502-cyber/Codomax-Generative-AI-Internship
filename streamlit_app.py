
# Chat With Your Documents - RAG Web App

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat With Your PDF", page_icon="📚")

st.title("📚 Chat With Your Documents")
st.write("Ask questions about your PDF document.")

# Read PDF
@st.cache_data
def load_pdf():
    pdf = PdfReader("documents/sample.pdf")
    text = ""

    for page in pdf.pages:
        text += page.extract_text() or ""

    return text

text = load_pdf()

# Split text into chunks
words = text.split()
chunk_size = 40

chunks = []

for i in range(0, len(words), chunk_size):
    chunks.append(" ".join(words[i:i + chunk_size]))

# Generate TF-IDF vectors
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

st.success("PDF loaded successfully!")
st.write("Total chunks:", len(chunks))

# Ask questions
question = st.text_input("Enter your question:")

if st.button("Search"):

    if question.strip():

        query_vector = vectorizer.transform([question])

        similarities = cosine_similarity(
            query_vector, chunk_vectors
        )

        best_index = similarities.argmax()
        score = similarities[0][best_index]

        if score > 0:
            st.subheader("Relevant Information")
            st.write(chunks[best_index])
            st.write("Similarity Score:", round(score, 3))
        else:
            st.warning("No relevant information found.")

    else:
        st.warning("Please enter a question.")
