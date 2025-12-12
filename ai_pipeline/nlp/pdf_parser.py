import fitz  # PyMuPDF
import re
import os

def extract_text_from_pdf(pdf_path):
    """
    PyMuPDF(fitz)를 사용하여 PDF에서 텍스트를 추출합니다.
    """
    doc = None
    full_text = ""
    
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            if text:
                full_text += text + "\n"
        
        # 전처리: 불필요한 공백, 특수문자 정리
        # (연속된 공백을 하나로 줄이고, 앞뒤 공백 제거)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return full_text

    except Exception as e:
        print(f" PDF 읽기 실패 ({os.path.basename(pdf_path)}): {e}")
        return None
    finally:
        if doc:
            doc.close()