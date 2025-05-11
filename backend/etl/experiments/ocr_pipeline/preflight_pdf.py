import fitz  # PyMuPDF

def is_pdf_text_based(filepath, min_text_threshold=100):
    doc = fitz.open(filepath)
    total_text = 0
    for page in doc:
        text = page.get_text()
        total_text += len(text.strip())
    doc.close()
    return total_text > min_text_threshold

# Example usage
pdf_path = "example.pdf"
if is_pdf_text_based(pdf_path):
    print("✅ This PDF is text-based. You can use Marker or Llamaparse.")
else:
    print("❌ This PDF is likely scanned. Run OCR first.")
