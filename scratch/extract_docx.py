import zipfile
import xml.etree.ElementTree as ET

def extract_text(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Find all paragraph elements and extract text
            paragraphs = []
            for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                text_elems = para.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                text = "".join([t.text for t in text_elems if t.text])
                if text:
                    paragraphs.append(text)
            
            return "\n".join(paragraphs)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    docx_path = r"d:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\docs\YOLO_TRAINING_REPORT_V1.docx"
    text = extract_text(docx_path)
    
    output_path = r"d:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\scratch\docx_extracted.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully extracted to docx_extracted.txt")
