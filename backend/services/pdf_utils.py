def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    from PyPDF2 import PdfReader
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text.strip()

