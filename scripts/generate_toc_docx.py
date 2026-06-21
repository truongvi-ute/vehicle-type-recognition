import docx
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_toc_document():
    doc = Document()
    
    # Page setup - Margins (standard 2.54 cm / 1 inch)
    for section in doc.sections:
        section.top_margin = docx.shared.Cm(2.5)
        section.bottom_margin = docx.shared.Cm(2.5)
        section.left_margin = docx.shared.Cm(3.0)
        section.right_margin = docx.shared.Cm(2.0)
        
    # Styles Setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(13)
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("MỤC LỤC CHI TIẾT BÁO CÁO CUỐI KỲ\n(TÍCH HỢP PHIÊN BẢN V3)")
    run_title.font.name = 'Times New Roman'
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0, 51, 102) # Dark Blue
    
    p_subtitle = doc.add_paragraph()
    p_subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_subtitle.add_run("Đồ án môn học: Xử lý ảnh số (252DIPR430685_07CLC)\nNhóm 04 - Nhận diện phương tiện giao thông")
    run_sub.font.name = 'Times New Roman'
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(12)
    
    toc_data = [
        # (Level, Title, Note, IsNewV3)
        (0, "LỜI CẢM ƠN", "", False),
        (0, "MỤC LỤC", "", False),
        (0, "MỤC LỤC HÌNH ẢNH", "", False),
        (0, "MỤC LỤC BẢNG BIỂU", "", False),
        
        (1, "1. Tổng quan đề tài", "", False),
        (2, "1.1. Bài toán, mục tiêu và ý nghĩa", "", False),
        (2, "1.2. Phạm vi, dữ liệu đầu vào và kết quả đầu ra", "", False),
        (2, "1.3. Tổng quan quy trình hệ thống", "Cập nhật sơ đồ luồng tích hợp V3", False),
        
        (1, "2. Bộ dữ liệu và chiến lược chuẩn bị dữ liệu", "", False),
        (2, "2.1. Nguồn dữ liệu", "", False),
        (2, "2.2. Mất cân bằng dữ liệu", "", False),
        (2, "2.3. Chuẩn bị và chia tập dữ liệu", "", False),
        
        (1, "3. Tiền xử lý và tăng cường dữ liệu", "", False),
        (2, "3.1. Quy trình tiền xử lý hình ảnh", "", False),
        (3, "3.1.1. Thay đổi kích thước ảnh giữ tỉ lệ và đệm viền đen (Zero-Padding)", "", False),
        (3, "3.1.2. Kỹ thuật đệm phản chiếu (Reflective Padding) - Cải tiến V3", "Cải tiến V3 - Thêm mới", True),
        (2, "3.2. Chiến lược cân bằng và mở rộng dữ liệu", "", False),
        (2, "3.3. Các phép biến đổi ảnh hình học và cường độ sáng", "", False),
        (2, "3.4. Phương pháp tăng cường dữ liệu V1 (Môi trường)", "", False),
        (2, "3.5. Phương pháp tăng cường dữ liệu V2 (Chất lượng ảnh)", "", False),
        (2, "3.6. Phương pháp tăng cường dữ liệu V3 (Tổ hợp tối ưu hóa nâng cao)", "Cải tiến V3 - Thêm mới", True),
        (2, "3.7. Tăng cường dữ liệu trực tuyến trong quá trình huấn luyện", "", False),
        (2, "3.8. Thiết kế các bộ dữ liệu thực nghiệm", "Thêm cleaned_v3 vào bảng đối sánh", False),
        (2, "3.9. Phân phối dữ liệu sau cân bằng và tăng cường", "", False),
        
        (1, "4. Mô hình và quá trình huấn luyện", "", False),
        (2, "4.1. Mạng tích chập thặng dư ResNet50", "", False),
        (3, "4.1.1. Tổng quan mô hình", "", False),
        (3, "4.1.2. Sơ đồ luồng xử lý của mô hình", "", False),
        (3, "4.1.3. Cấu hình huấn luyện baseline (V1/V2)", "", False),
        (3, "4.1.4. Cấu hình cải tiến V3: Tích hợp Loss Fusion & Cosine Decay", "Cải tiến V3 - Thêm mới", True),
        (2, "4.2. Mạng YOLO Classification", "", False),
        (3, "4.2.1. Tổng quan mô hình (Nâng cấp từ YOLOv8n-cls lên YOLOv8m-cls)", "Cập nhật V3", False),
        (3, "4.2.2. Sơ đồ luồng xử lý của mô hình", "", False),
        (3, "4.2.3. Cấu hình huấn luyện baseline (V1/V2)", "", False),
        (3, "4.2.4. Cấu hình cải tiến V3: Nâng cấp YOLOv8m-cls & Confused Pair Margin Loss", "Cải tiến V3 - Thêm mới", True),
        (2, "4.3. Kiến trúc Vision Transformer (ViT)", "", False),
        (3, "4.3.1. Tổng quan mô hình", "", False),
        (3, "4.3.2. Sơ đồ luồng xử lý của mô hình", "", False),
        (3, "4.3.3. Cấu hình huấn luyện baseline (V1/V2)", "", False),
        (3, "4.3.4. Cấu hình cải tiến V3: Tối ưu hóa Fine-tuning & Loss Fusion", "Cải tiến V3 - Thêm mới", True),
        
        (1, "5. Kết quả và đánh giá", "", False),
        (2, "5.1. Các tiêu chí đánh giá mô hình", "", False),
        (2, "5.2. Kết quả huấn luyện của ResNet50", "", False),
        (3, "5.2.1. Biểu đồ Loss (So sánh V1, V2 và V3)", "Cập nhật V3", False),
        (3, "5.2.2. Biểu đồ Validation Accuracy (So sánh V1, V2 và V3)", "Cập nhật V3", False),
        (3, "5.2.3. Ma trận nhầm lẫn (Confusion Matrix) của ResNet50 V3", "Cập nhật V3", False),
        (3, "5.2.4. Chỉ số Precision, Recall và F1-score theo từng lớp của ResNet50 V3", "Cập nhật V3", False),
        (3, "5.2.5. Phân tích hiệu năng cải tiến V3 trên ResNet50", "Cải tiến V3 - Thêm mới", True),
        (2, "5.3. Kết quả huấn luyện của YOLO Classification", "", False),
        (3, "5.3.1. Biểu đồ Loss (So sánh V1, V2 và V3)", "Cập nhật V3", False),
        (3, "5.3.2. Biểu đồ Validation Accuracy (So sánh V1, V2 và V3)", "Cập nhật V3", False),
        (3, "5.3.3. Ma trận nhầm lẫn (Confusion Matrix) của YOLO V3", "Cập nhật V3", False),
        (3, "5.3.4. Chỉ số Precision, Recall và F1-score theo từng lớp của YOLO V3", "Cập nhật V3", False),
        (3, "5.3.5. Phân tích hiệu năng cải tiến V3 trên YOLOv8m-cls", "Cải tiến V3 - Thêm mới", True),
        (2, "5.4. Kết quả huấn luyện của Vision Transformer (ViT)", "", False),
        (3, "5.4.1. Biểu đồ Loss (So sánh V1, V2 và V3)", "Cập nhật V3", False),
        (3, "5.4.2. Biểu đồ Validation Accuracy (So sánh V1, V2 và V3)", "Cập nhật V3", False),
        (3, "5.4.3. Ma trận nhầm lẫn (Confusion Matrix) của ViT V3", "Cập nhật V3", False),
        (3, "5.4.4. Chỉ số Precision, Recall và F1-score theo từng lớp của ViT V3", "Cập nhật V3", False),
        (3, "5.4.5. Phân tích hiệu năng cải tiến V3 trên Vision Transformer", "Cải tiến V3 - Thêm mới", True),
        (2, "5.5. So sánh kết quả thực nghiệm giữa ba mô hình", "", False),
        (3, "5.5.1. Phân tích các lớp nhận diện ưu việt", "", False),
        (3, "5.5.2. Phân tích các lớp nhạy cảm và hướng cải thiện", "", False),
        (3, "5.5.3. Nhận xét tổng quan về kết quả thực nghiệm (V1 vs V2 vs V3)", "Cập nhật V3", False),
        (3, "5.5.4. Chứng minh thực nghiệm hiệu quả của Reflective Padding & Loss Fusion", "Cải tiến V3 - Thêm mới", True),
        
        (1, "6. Kết luận và hướng phát triển", "", False),
        (2, "6.1. Cấu hình mô hình tối ưu đề xuất", "", False),
        (3, "6.1.1. Định hướng tối ưu hóa độ chính xác (ViT V3)", "Cải tiến V3 - Thêm mới", True),
        (3, "6.1.2. Định hướng tối ưu hóa tốc độ thời gian thực (YOLOv8m V3)", "Cải tiến V3 - Thêm mới", True),
        (2, "6.2. Hạn chế", "", False),
        (2, "6.3. Hướng phát triển", "", False)
    ]
    
    for level, title, note, is_new in toc_data:
        p = doc.add_paragraph()
        p_format = p.paragraph_format
        
        # Indentation and spacing based on heading level
        if level == 0:
            p_format.left_indent = docx.shared.Inches(0.0)
            p_format.space_before = Pt(8)
            p_format.space_after = Pt(4)
            run = p.add_run(title.upper())
            run.font.bold = True
            run.font.size = Pt(13)
        elif level == 1:
            p_format.left_indent = docx.shared.Inches(0.0)
            p_format.space_before = Pt(12)
            p_format.space_after = Pt(6)
            run = p.add_run(title)
            run.font.bold = True
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(0, 51, 102) # Dark Blue
        elif level == 2:
            p_format.left_indent = docx.shared.Inches(0.3)
            p_format.space_before = Pt(4)
            p_format.space_after = Pt(2)
            run = p.add_run(title)
            if is_new:
                run.font.bold = True
            run.font.size = Pt(12)
        elif level == 3:
            p_format.left_indent = docx.shared.Inches(0.6)
            p_format.space_before = Pt(2)
            p_format.space_after = Pt(2)
            run = p.add_run(title)
            run.font.italic = not is_new
            if is_new:
                run.font.bold = True
            run.font.size = Pt(11)
            
        # Style the V3 / Updates labels
        if note:
            run_note = p.add_run(f" ({note})")
            run_note.font.size = Pt(10)
            if is_new:
                run_note.font.bold = True
                run_note.font.color.rgb = RGBColor(180, 0, 0) # Dark Red for new V3 items
            else:
                run_note.font.italic = True
                run_note.font.color.rgb = RGBColor(128, 128, 128) # Grey for baseline updates
                
    output_path = "docs/MucLuc_BaoCao_V3.docx"
    doc.save(output_path)
    print(f"Document successfully saved to {output_path}")

if __name__ == '__main__':
    create_toc_document()
