from io import BytesIO
import pypdf

class PDFExtractor:
    @staticmethod
    def extract_text(file_content: bytes):
        """
        Extracts text from a PDF file content.
        Returns a list of (page_number, text) tuples.
        """
        pdf_file = BytesIO(file_content)
        reader = pypdf.PdfReader(pdf_file)
        
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_pages.append({"page": i + 1, "text": text})
            else:
                extracted_pages.append({"page": i + 1, "text": ""})
        
        return extracted_pages

    @staticmethod
    def is_scanned(extracted_pages, threshold=50):
        """
        Heuristic to check if PDF is scanned.
        If minimal text is extracted, it's likely scanned.
        """
        total_text_len = sum(len(p["text"].strip()) for p in extracted_pages)
        # Verify if average characters per page is below threshold
        if not extracted_pages:
             return True
        avg_chars = total_text_len / len(extracted_pages)
        return avg_chars < threshold
