
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample sentences
sentences = [
    "I love learning Python programming",
    "Python is my favourite programming language",
    "I enjoy playing cricket"
]

# Convert text into numerical embeddings
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(sentences)

# Calculate semantic similarity
similarity = cosine_similarity(embeddings)

print("Text Embeddings:")
print(embeddings.toarray())

print("\nSimilarity Scores:")
print(similarity)

print("\nComparison:")
print("Sentence 1 and Sentence 2:",
      similarity[0][1])
