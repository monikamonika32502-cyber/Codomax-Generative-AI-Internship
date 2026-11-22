
# Day 18 - PDF Document Processing

from pypdf import PdfReader

# Read PDF from documents folder
pdf = PdfReader("documents/sample.pdf")

# Extract text
text = ""

for page in pdf.pages:
    text += page.extract_text() or ""

print("PDF Content:")
print(text)

# Count words
words = text.split()

print("\nTotal Words:", len(words))
print("\nPDF Processing Completed!")

