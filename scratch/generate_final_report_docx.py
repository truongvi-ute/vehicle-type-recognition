# -*- coding: utf-8 -*-
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import os

def set_cell_background(cell, hex_color):
    """Sets background color of a cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    """Sets cell margins (padding) in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Applies clean horizontal borders and removes vertical borders."""
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(tblBorders)

def add_callout_box(doc, text, title="LƯU Ý QUAN TRỌNG"):
    """Adds a stylish callout box with a thick left border and light shading."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=160, bottom=160, left=240, right=240)
    
    # Custom left border (deep navy) and remove others
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"★ {title}\n")
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(11)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(27, 54, 93)
    
    run_text = p.add_run(text)
    run_text.font.name = 'Arial'
    run_text.font.size = Pt(10)
    run_text.font.italic = True
    run_text.font.color.rgb = RGBColor(80, 80, 80)
    
    # Empty paragraph for spacing
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(6)
    p_after.paragraph_format.space_after = Pt(6)

def style_text_run(run, font_name="Arial", size_pt=11, bold=False, italic=False, color_rgb=None):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb

def create_report():
    doc = Document()
    
    # Page Margins (1 inch on all sides)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
        # Configure Header & Footer
        section.different_first_page_header_footer = True
        
        # Header (Page 2 onwards)
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Đề Xuất Cải Tiến Trích Xuất Đặc Trưng & Xử Lý Ảnh | Dự Án Nhóm")
        style_text_run(hrun, font_name="Arial", size_pt=8.5, color_rgb=RGBColor(120, 120, 120))
        
        # Footer (Page 2 onwards)
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Trang 2 / 5 (Tài liệu báo cáo kỹ thuật chuyên sâu)")
        style_text_run(frun, font_name="Arial", size_pt=8.5, color_rgb=RGBColor(120, 120, 120))
        
    # Palettes
    NAVY = RGBColor(27, 54, 93)      # #1B365D - Primary
    TEAL = RGBColor(46, 91, 112)     # #2E5B70 - Secondary
    CHARCOAL = RGBColor(51, 51, 51)  # #333333 - Body text
    MUTED = RGBColor(120, 120, 120)  # #787878 - Subtitles/Notes
    
    # ------------------ DOCUMENT TITLE ------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    p_title.paragraph_format.space_after = Pt(6)
    t_run = p_title.add_run("BẢN ĐỀ XUẤT CẢI TIẾN CHUYÊN NGHIỆP\nNÂNG CAO HIỆU NĂNG HỆ THỐNG NHẬN DIỆN PHƯƠNG TIỆN")
    style_text_run(t_run, font_name="Arial", size_pt=18, bold=True, color_rgb=NAVY)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    s_run = p_sub.add_run("Phân tích suy luận kỹ thuật từ số liệu thực tế của 3 mô hình (ResNet-50, ViT, YOLOv8-cls) và phân nhóm lộ trình cải tiến theo luồng huấn luyện")
    style_text_run(s_run, font_name="Arial", size_pt=11, italic=True, color_rgb=TEAL)
    
    # Horizontal line
    p_sep = doc.add_paragraph()
    p_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sep.paragraph_format.space_after = Pt(20)
    sep_run = p_sep.add_run("—" * 60)
    style_text_run(sep_run, font_name="Arial", size_pt=10, color_rgb=MUTED)
    
    # ------------------ INTRODUCTION ------------------
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.15
    p_intro.paragraph_format.space_after = Pt(12)
    irun = p_intro.add_run(
        "Tài liệu này trình bày các phân tích khoa học sâu sắc dựa trên dữ liệu thu được từ quá trình kiểm thử "
        "thực tế 3 mô hình học máy: YOLOv8-cls, ResNet-50 và Vision Transformer (ViT-Base) trên tập Test Split (3,601 ảnh). "
        "Dựa vào các điểm nghẽn nhận diện đã được bóc tách định lượng, đề xuất này đưa ra các giải pháp cụ thể "
        "nhằm tái cấu trúc luồng tiền xử lý ảnh và cơ chế trích xuất đặc trưng của mô hình, được phân loại rõ ràng "
        "theo luồng huấn luyện (In-Training) và ngoài luồng huấn luyện (Out-of-Training) để triển khai thực tế."
    )
    style_text_run(irun, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # ------------------ SECTION I ------------------
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(8)
    h1_run = h1.add_run("I. THỰC TRẠNG VÀ PHÂN TÍCH SUY LUẬN SAI SỐ HỆ THỐNG")
    style_text_run(h1_run, font_name="Arial", size_pt=13, bold=True, color_rgb=NAVY)
    
    p_desc1 = doc.add_paragraph()
    p_desc1.paragraph_format.space_after = Pt(10)
    p_desc1.paragraph_format.line_spacing = 1.15
    run_desc1 = p_desc1.add_run(
        "Hiệu năng của các mô hình trên tập kiểm thử độc lập được thống kê dưới dạng chỉ số chung (Bảng 1) "
        "và chỉ số chi tiết cho từng lớp (Bảng 2). Từ đó, nhóm áp dụng suy luận kỹ thuật để bóc tách 3 điểm nghẽn sai số cốt lõi."
    )
    style_text_run(run_desc1, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # Table 1: Model Comparison
    t1 = doc.add_table(rows=5, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)
    
    headers_t1 = ["Chỉ số đo lường", "YOLOv8-cls (Lightweight)", "ResNet-50 (Baseline CNN)", "Vision Transformer (ViT-Base)"]
    col_widths_t1 = [Inches(2.3), Inches(1.4), Inches(1.4), Inches(1.4)]
    
    for idx, name in enumerate(headers_t1):
        cell = t1.cell(0, idx)
        cell.width = col_widths_t1[idx]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(name)
        style_text_run(run, font_name="Arial", size_pt=9.5, bold=True, color_rgb=RGBColor(255, 255, 255))
        
    data_t1 = [
        ["Độ chính xác (Accuracy)", "91.53%", "91.70%", "94.06%"],
        ["Hàm mất mát (Loss)", "0.2618", "0.7199", "0.6557"],
        ["Macro F1-Score", "89.18%", "89.86%", "92.31%"],
        ["Weighted F1-Score", "91.45%", "91.70%", "94.05%"]
    ]
    
    for row_idx, row_data in enumerate(data_t1, start=1):
        bg_color = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            cell = t1.cell(row_idx, col_idx)
            cell.width = col_widths_t1[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            
            is_vit_highest = (col_idx == 3 and row_idx in [1, 3, 4])
            style_text_run(run, font_name="Arial", size_pt=9.5, bold=is_vit_highest, 
                           color_rgb=NAVY if is_vit_highest else CHARCOAL)
            
    p_caption1 = doc.add_paragraph()
    p_caption1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_caption1.paragraph_format.space_before = Pt(6)
    p_caption1.paragraph_format.space_after = Pt(16)
    run_cap1 = p_caption1.add_run("Bảng 1: So sánh hiệu năng tổng thể của các mô hình")
    style_text_run(run_cap1, font_name="Arial", size_pt=9, italic=True, color_rgb=MUTED)
    
    # Table 2: Class F1
    t2 = doc.add_table(rows=11, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)
    
    headers_t2 = ["Lớp phương tiện", "Số mẫu hỗ trợ (Support)", "YOLOv8-cls F1", "ResNet-50 F1", "ViT-Base F1"]
    col_widths_t2 = [Inches(2.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1)]
    
    for idx, name in enumerate(headers_t2):
        cell = t2.cell(0, idx)
        cell.width = col_widths_t2[idx]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(name)
        style_text_run(run, font_name="Arial", size_pt=9.5, bold=True, color_rgb=RGBColor(255, 255, 255))
        
    data_t2 = [
        ["bicycle", "162", "92.7%", "92.1%", "93.8%"],
        ["boat", "890", "96.4%", "97.1%", "98.3%"],
        ["bus", "406", "93.1%", "93.0%", "95.5%"],
        ["car", "854", "89.7%", "89.4%", "92.2%"],
        ["helicopter", "67", "93.0%", "95.4%", "96.3%"],
        ["minibus", "148", "79.9%", "81.3%", "80.6%"],
        ["motorcycle", "444", "96.5%", "96.2%", "98.0%"],
        ["taxi", "91", "71.6%", "75.4%", "81.3%"],
        ["train", "168", "97.0%", "96.2%", "99.7%"],
        ["truck", "371", "82.0%", "82.6%", "87.6%"]
    ]
    
    for row_idx, row_data in enumerate(data_t2, start=1):
        bg_color = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            cell = t2.cell(row_idx, col_idx)
            cell.width = col_widths_t2[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            
            is_low_f1 = (col_idx >= 2 and float(text.replace('%', '')) < 85.0)
            if is_low_f1:
                style_text_run(run, font_name="Arial", size_pt=9.5, bold=True, color_rgb=RGBColor(180, 40, 40))
            else:
                style_text_run(run, font_name="Arial", size_pt=9.5, color_rgb=CHARCOAL)
                
    p_caption2 = doc.add_paragraph()
    p_caption2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_caption2.paragraph_format.space_before = Pt(6)
    p_caption2.paragraph_format.space_after = Pt(16)
    run_cap2 = p_caption2.add_run("Bảng 2: F1-Score chi tiết từng lớp trên tập Test")
    style_text_run(run_cap2, font_name="Arial", size_pt=9, italic=True, color_rgb=MUTED)
    
    # 3 Error Analyses
    analyses = [
        ("1. Lỗi Taxi \u2192 Car: Bài toán 'Tần số không gian' (Spatial Frequency)",
         " Số liệu chỉ ra lỗi nghiêm trọng ở nhóm taxi (91 mẫu). YOLOv8-cls bỏ sót 33 ảnh (36.3%), ResNet-50 nhầm 24 ảnh (26.4%), ViT nhầm 13 ảnh (14.3%) sang Car. "
         "Về mặt vật lý, phom dáng tổng thể của taxi và xe con hoàn toàn trùng khớp. Điểm phân biệt duy nhất nằm ở mào taxi trên nóc xe (chiếm tỷ lệ diện tích dưới 2% toàn bức ảnh). "
         "Đối với mô hình CNN (ResNet, YOLO), khi ảnh nạp vào ở kích thước 224x224, qua các lớp downsampling tích lũy tiến, kích thước feature map ở block cuối chỉ còn 7x7. "
         "Lúc này, đặc trưng chiếc mào xe (vốn có tần số không gian rất cao và kích thước nhỏ) bị hòa tan hoàn toàn vào pixel xung quanh, khiến mô hình chỉ nhận diện được đặc trưng thân xe ô tô chung chung. "
         "Ngược lại, cơ chế Self-Attention của ViT duy trì độ phân giải patch 14x14 không đổi qua các lớp, cho phép CLS token trực tiếp truy vấn đặc trưng cục bộ này."),
         
        ("2. Lỗi Minibus \u2192 Car/Truck: Bài toán 'Mối quan hệ không gian' (Spatial Relationship)",
         " Lớp minibus có F1-Score cực thấp trên cả 3 mô hình (79.9% - 81.3%). Các mô hình nhầm minibus sang cả 2 phía: Car (SUV lớn) và Truck (xe tải nhỏ). "
         "Nguyên nhân vật lý là xe khách 16 chỗ (Minibus) có tỷ lệ hình khối chữ nhật và các góc quan sát lấp lửng giữa 2 nhóm trên. "
         "Cơ chế Global Average Pooling (GAP) mặc định tính trung bình cộng toàn bộ đặc trưng không gian, vô tình cào bằng và triệt tiêu các đặc trưng quan trọng như: tỷ lệ phân bổ chiều dài thân xe, "
         "và cấu trúc dạng lưới của dãy cửa kính hông (đặc trưng phân biệt cốt lõi của minibus so với xe bán tải hay xe hơi 7 chỗ)."),
         
        ("3. Lỗi Truck \u2192 Car/Bus: Bài toán 'Độ tương phản biên dạng' (Edge Contrast Degradation)",
         " Recall của truck ở YOLOv8-cls chỉ đạt 80.1% (nhầm 37 mẫu sang car, 15 sang bus). Sai số xảy ra nặng nề nhất trên tập ảnh chụp ban đêm (night) hoặc trời mưa (rain). "
         "Trong điều kiện ánh sáng yếu hoặc mờ nhòe do hạt mưa, các bộ lọc tích chập ở các tầng nông (vốn được thiết kế để trích xuất biên thùng xe tải vuông vức) không thể kích hoạt (activation = 0) do gradient độ sáng quá thấp. "
         "Mô hình mất đi đặc trưng 'thùng xe góc vuông' và chỉ bắt được hình dáng cabin đầu xe bo tròn, dẫn đến chẩn đoán nhầm thành Car hoặc Bus.")
    ]
    
    for title, body in analyses:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        
        rt = p.add_run(title + "\n")
        style_text_run(rt, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
        
        rb = p.add_run(body)
        style_text_run(rb, font_name="Arial", size_pt=10, color_rgb=CHARCOAL)
        
    # ------------------ SECTION II ------------------
    h2 = doc.add_heading(level=1)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(8)
    h2_run = h2.add_run("II. BẢNG TỔNG HỢP ĐỀ XUẤT CẢI TIẾN HỆ THỐNG (TRAIN VS OUT-OF-TRAIN)")
    style_text_run(h2_run, font_name="Arial", size_pt=13, bold=True, color_rgb=NAVY)
    
    p_desc2 = doc.add_paragraph()
    p_desc2.paragraph_format.space_after = Pt(10)
    p_desc2.paragraph_format.line_spacing = 1.15
    run_desc2 = p_desc2.add_run(
        "Nhằm tối ưu hóa luồng triển khai, các phương pháp cải tiến được phân tách rõ ràng thành hai nhóm: "
        "Cải tiến trực tiếp trong quá trình huấn luyện (In-Training) để tinh chỉnh mạng, và Cải tiến ở các luồng ngoài huấn luyện (Out-of-Training) "
        "như tiền xử lý dữ liệu và hậu xử lý suy luận."
    )
    style_text_run(run_desc2, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # 6-column Table (STT, Phân nhóm, Phương pháp, Kỹ thuật, Dẫn chứng, Cơ sở)
    # Total printable width is 6.5 inches.
    t3 = doc.add_table(rows=8, cols=6)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3, color="B0C4DE", sz="6")
    
    headers_t3 = [
        "STT", 
        "Phân Nhóm Luồng", 
        "Phương Pháp Đề Xuất", 
        "Giải Pháp Chi Tiết & Kỹ Thuật", 
        "Dẫn Chứng Cần Khắc Phục", 
        "Cơ Chế Hoạt Động & Cơ Sở Khoa Học"
    ]
    col_widths_t3 = [Inches(0.35), Inches(0.95), Inches(1.25), Inches(1.45), Inches(1.15), Inches(1.35)]
    
    for idx, name in enumerate(headers_t3):
        cell = t3.cell(0, idx)
        cell.width = col_widths_t3[idx]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=120, bottom=120, left=60, right=60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(name)
        style_text_run(run, font_name="Arial", size_pt=8.5, bold=True, color_rgb=RGBColor(255, 255, 255))
        
    data_t3 = [
        [
            "1",
            "Ngoài Train\n(Tiền xử lý)",
            "Reflective Padding\n(Đệm đối xứng)",
            "Thay thế zero_pad_to_square bằng kỹ thuật lấy đối xứng biên ảnh qua gương (Mirror/Reflective padding) khi chuẩn bị tập dữ liệu.",
            "Biên ảnh chuyển đột ngột từ ảnh thực sang viền đen tạo ra các cạnh sắc nhọn (xung nhiễu tần số cao) làm rối loạn bộ lọc tích chập.",
            "Triệt tiêu nhiễu tần số cao tại biên. Giúp phân phối giá trị trung bình (mean activation map) ổn định hơn, hướng sự tập trung (Attention) vào phương tiện."
        ],
        [
            "2",
            "Ngoài Train\n(Tiền xử lý)",
            "LAB-CLAHE\n(Cân bằng sáng)",
            "Chuyển ảnh sang không gian màu LAB, áp dụng CLAHE trên kênh L (Luminance) với ngưỡng giới hạn tương phản (clip limit = 2.0).",
            "Ảnh thời tiết ngoại tuyến (night chiếm 10%, rain chiếm 10%) làm mờ biên dạng xe tải của Truck (YOLO nhầm 37 car, 15 bus).",
            "Tăng cường độ tương phản cục bộ mà không gây cháy sáng hay khuếch đại nhiễu hạt. Giúp làm nổi rõ biên của xe tải trong bóng tối."
        ],
        [
            "3",
            "Trong Train\n(Hàm Loss)",
            "Class-Balanced Focal Loss",
            "Thay thế Cross-Entropy bằng Focal Loss kết hợp trọng số nghịch đảo mẫu hiệu dụng: W_i = (1 - \u03b2) / (1 - \u03b2^{n_i})",
            "Lớp taxi (91 mẫu), helicopter (67 mẫu) bị thiên lệch nghiêm trọng về lớp đa số boat (890 mẫu) và car (854 mẫu).",
            "Focal Loss hạ trọng số của mẫu dễ (Boat, Car), tăng gradient mẫu sai. Class-Balanced Weighting định hình lại không gian tối ưu cho lớp thiểu số."
        ],
        [
            "4",
            "Trong Train\n(Augmentation)",
            "ROI-Focused Random Erasing",
            "Thực hiện che khuất ngẫu nhiên (Random Erasing) giới hạn tọa độ (chỉ áp dụng ở nửa trên cabin xe con và xe tải) khi sinh batch.",
            "33 Taxi nhầm thành Car và 37 Truck nhầm thành Car do mô hình chỉ học các đặc trưng dễ ăn điểm (bánh xe, đầu xe).",
            "Buộc mô hình học đặc trưng phân tán (Distributed Representations). Khi cabin bị che, mô hình buộc phải học đèn pha, lưới tản nhiệt, đuôi xe."
        ],
        [
            "5",
            "Trong Train\n(Kiến trúc ResNet)",
            "Trích xuất đặc trưng đa quy mô (ResNet-50)",
            "Subclass ResNet-50, thực hiện Average Pooling trên Layer 3 (1024ch) và Layer 4 (2048ch), concatenate thành vector 3072ch nạp vào head.",
            "Recall Taxi cực thấp ở CNN (YOLO nhầm 36.3%, ResNet nhầm 26.4%) do đặc trưng mào xe bị tiêu biến ở Layer 4 (strides = 32).",
            "Layer 3 (strides = 16) lưu giữ các đặc trưng hình học tần số cao cục bộ chưa bị nén quá mức. Layer 4 cung cấp bối cảnh ngữ nghĩa toàn cục."
        ],
        [
            "6",
            "Trong Train\n(Kiến trúc ViT)",
            "Multi-Layer CLS Token Fusion",
            "Sử dụng PyTorch Forward Hook trích xuất CLS Token ở Layer 10, 11 và 12, concatenate thành vector 2304ch trước khi đưa vào classifier.",
            "Minibus có F1-score thấp (~80.6% trên ViT) do token CLS cuối cùng bị mịn hóa quá mức, mất thông tin cấu trúc hông xe.",
            "CLS Token ở các block giữa lưu trữ thông tin hình thái và tỉ lệ thô của xe, ghép với block cuối giúp ổn định ranh giới quyết định không gian."
        ],
        [
            "7",
            "Ngoài Train\n(Suy luận/Inference)",
            "Ensemble Soft-Voting",
            "Tích hợp kết quả suy luận của 3 mô hình trên Flask API theo trọng số: P = 0.5*P(ViT) + 0.3*P(Res50) + 0.2*P(YOLO).",
            "ViT chính xác cao (94.06%) nhưng chậm. YOLO cực nhanh (50ms) nhưng Recall Taxi yếu (63.7%).",
            "Kết hợp thế mạnh của 2 trường phái: Global Attention (ViT học bối cảnh tốt) và Local Inductive Bias (ResNet/YOLO học biên tốt). Giảm thiểu lỗi cá biệt."
        ]
    ]
    
    for row_idx, row_data in enumerate(data_t3, start=1):
        bg_color = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            cell = t3.cell(row_idx, col_idx)
            cell.width = col_widths_t3[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=40, right=40)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            
            if col_idx in [0, 1]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            run = p.add_run(text)
            
            # Format text formatting
            is_bold = (col_idx in [0, 1, 2])
            color = NAVY if is_bold else CHARCOAL
            if col_idx == 1:
                # Highlight training vs non-training with colors
                if "Trong Train" in text:
                    color = RGBColor(160, 40, 40) # Dark Red for training
                else:
                    color = RGBColor(40, 100, 40) # Dark Green for others
                    
            style_text_run(run, font_name="Arial", size_pt=8.5, bold=is_bold, color_rgb=color)
            
    p_caption3 = doc.add_paragraph()
    p_caption3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_caption3.paragraph_format.space_before = Pt(6)
    p_caption3.paragraph_format.space_after = Pt(16)
    run_cap3 = p_caption3.add_run("Bảng 3: Ma trận giải pháp cải tiến phân nhóm theo luồng huấn luyện")
    style_text_run(run_cap3, font_name="Arial", size_pt=9, italic=True, color_rgb=MUTED)
    
    # ------------------ SECTION III ------------------
    h3 = doc.add_heading(level=1)
    h3.paragraph_format.space_before = Pt(18)
    h3.paragraph_format.space_after = Pt(8)
    h3_run = h3.add_run("III. CƠ SỞ TOÁN HỌC VÀ THIẾT KẾ KIẾN TRÚC MÔ HÌNH")
    style_text_run(h3_run, font_name="Arial", size_pt=13, bold=True, color_rgb=NAVY)
    
    p_eq_desc = doc.add_paragraph()
    p_eq_desc.paragraph_format.space_after = Pt(10)
    p_eq_desc.paragraph_format.line_spacing = 1.15
    run_eq_desc = p_eq_desc.add_run(
        "Dưới đây là đặc tả toán học của hàm lỗi cải tiến và mã nguồn cấu trúc trích xuất đặc trưng mới đề xuất."
    )
    style_text_run(run_eq_desc, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # Mathematical explanation
    p_math = doc.add_paragraph()
    p_math.paragraph_format.left_indent = Inches(0.25)
    p_math.paragraph_format.space_after = Pt(12)
    p_math.paragraph_format.line_spacing = 1.2
    
    m_title1 = p_math.add_run("1. Hàm mất mát Class-Balanced Focal Loss:\n")
    style_text_run(m_title1, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
    
    m_eq1 = p_math.add_run("   L_{CB-Focal}(p_t) = - W_i \u22c5 (1 - p_t)^\u03b3 \u22c5 log(p_t)\n")
    style_text_run(m_eq1, font_name="Courier New", size_pt=11, bold=True, color_rgb=NAVY)
    
    m_eq1_sub = p_math.add_run("   Trong đó: W_i = (1 - \u03b2) / (1 - \u03b2^{n_i})  (với \u03b2 = 0.999, \u03b3 = 2.0)\n\n")
    style_text_run(m_eq1_sub, font_name="Courier New", size_pt=10, color_rgb=NAVY)
    
    m_desc = p_math.add_run(
        "Cơ chế: W_i tái cấu trúc gradient để cân bằng ảnh hưởng của số lượng mẫu thực tế n_i. "
        "Đại lượng (1 - p_t)^\u03b3 triệt tiêu triệt để hàm lỗi từ các mẫu dễ (Boat, Car với p_t \u2192 1), "
        "đồng thời phóng đại gradient từ các mẫu khó phân loại (Taxi, Minibus với p_t thấp)."
    )
    style_text_run(m_desc, font_name="Arial", size_pt=10, color_rgb=CHARCOAL)
    
    # ResNet-50 code snippet
    p_code1 = doc.add_paragraph()
    p_code1.paragraph_format.space_before = Pt(8)
    p_code1.paragraph_format.space_after = Pt(6)
    r_code1_title = p_code1.add_run("2. Triển khai trích xuất đặc trưng đa quy mô cho ResNet-50 (PyTorch):")
    style_text_run(r_code1_title, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
    
    # Single-cell table for code formatting
    t_code1 = doc.add_table(rows=1, cols=1)
    t_code1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_background(t_code1.cell(0, 0), "F6F8FA")
    set_cell_margins(t_code1.cell(0, 0), top=100, bottom=100, left=150, right=150)
    
    code_resnet = (
        "class MultiScaleResNet50(nn.Module):\n"
        "    def __init__(self, original_resnet, num_classes=10):\n"
        "        super().__init__()\n"
        "        self.backbone_low = nn.Sequential(*list(original_resnet.children())[:-3]) # Layer3 và trước đó\n"
        "        self.layer4 = original_resnet.layer4\n"
        "        self.avgpool = original_resnet.avgpool\n"
        "        # Ghép đặc trưng Layer 3 (1024ch) + Layer 4 (2048ch)\n"
        "        self.fc = nn.Sequential(\n"
        "            nn.Linear(1024 + 2048, 512),\n"
        "            nn.BatchNorm1d(512),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.Dropout(0.3),\n"
        "            nn.Linear(512, num_classes)\n"
        "        )\n\n"
        "    def forward(self, x):\n"
        "        f3 = self.backbone_low(x)       # [B, 1024, 14, 14]\n"
        "        f4 = self.layer4(f3)            # [B, 2048, 7, 7]\n"
        "        p3 = torch.flatten(self.avgpool(f3), 1)\n"
        "        p4 = torch.flatten(self.avgpool(f4), 1)\n"
        "        fused = torch.cat([p3, p4], dim=1) # [B, 3072]\n"
        "        return self.fc(fused)"
    )
    p_c1 = t_code1.cell(0, 0).paragraphs[0]
    p_c1.paragraph_format.line_spacing = 1.05
    run_c1 = p_c1.add_run(code_resnet)
    style_text_run(run_c1, font_name="Courier New", size_pt=9, color_rgb=CHARCOAL)
    
    # ViT code snippet
    p_code2 = doc.add_paragraph()
    p_code2.paragraph_format.space_before = Pt(12)
    p_code2.paragraph_format.space_after = Pt(6)
    r_code2_title = p_code2.add_run("3. Triển khai trích xuất CLS Token đa lớp cho Vision Transformer (PyTorch):")
    style_text_run(r_code2_title, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
    
    t_code2 = doc.add_table(rows=1, cols=1)
    t_code2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_background(t_code2.cell(0, 0), "F6F8FA")
    set_cell_margins(t_code2.cell(0, 0), top=100, bottom=100, left=150, right=150)
    
    code_vit = (
        "class MultiLayerCLSViT(nn.Module):\n"
        "    def __init__(self, original_vit, num_classes=10, num_layers=3):\n"
        "        super().__init__()\n"
        "        self.vit = original_vit\n"
        "        self.vit.heads = nn.Identity() # Bỏ head mặc định\n"
        "        self.fc = nn.Linear(768 * num_layers, num_classes)\n"
        "        self.tokens = []\n"
        "        # Gắn hooks vào các lớp encoder cuối cùng\n"
        "        total = len(self.vit.encoder.layers)\n"
        "        for i in range(total - num_layers, total):\n"
        "            self.vit.encoder.layers[i].register_forward_hook(self._hook())\n\n"
        "    def _hook(self):\n"
        "        def h_fn(module, input, output):\n"
        "            self.tokens.append(output[:, 0]) # Trích xuất CLS Token\n"
        "        return h_fn\n\n"
        "    def forward(self, x):\n"
        "        self.tokens.clear()\n"
        "        _ = self.vit(x)\n"
        "        fused = torch.cat(self.tokens, dim=1) # Ghép các CLS Tokens\n"
        "        return self.fc(fused)"
    )
    p_c2 = t_code2.cell(0, 0).paragraphs[0]
    p_c2.paragraph_format.line_spacing = 1.05
    run_c2 = p_c2.add_run(code_vit)
    style_text_run(run_c2, font_name="Courier New", size_pt=9, color_rgb=CHARCOAL)
    
    # Forcing spacing
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(8)
    
    # ------------------ SECTION IV ------------------
    h4 = doc.add_heading(level=1)
    h4.paragraph_format.space_before = Pt(18)
    h4.paragraph_format.space_after = Pt(8)
    h4_run = h4.add_run("IV. LỘ TRÌNH TRIỂN KHAI VÀ CAM KẾT CHẤT LƯỢNG")
    style_text_run(h4_run, font_name="Arial", size_pt=13, bold=True, color_rgb=NAVY)
    
    p_desc4 = doc.add_paragraph()
    p_desc4.paragraph_format.space_after = Pt(10)
    p_desc4.paragraph_format.line_spacing = 1.15
    run_desc4 = p_desc4.add_run(
        "Kế hoạch triển khai được thực thi song song trên cả hai luồng để tối ưu hóa thời gian phát triển:"
    )
    style_text_run(run_desc4, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # Bullet points
    timeline = [
        ("1. Cải tiến Luồng Ngoài Huấn Luyện (Tuần 1):",
         " Tái cấu trúc pipeline tiền xử lý ảnh để áp dụng Reflective Padding và LAB-CLAHE. Module này được đóng gói độc lập trong backend API và data loader để đảm bảo tính nhất quán giữa huấn luyện và suy luận trực tuyến."),
        ("2. Cải tiến Luồng Trong Huấn Luyện (Tuần 2-3):",
         " Triển khai các lớp Wrapper trích xuất đặc trưng mới cho ResNet-50 và ViT trong file model.py. Tích hợp Class-Balanced Focal Loss vào module huấn luyện trên Kaggle và chạy huấn luyện lại hệ thống."),
        ("3. Kiểm thử và Đánh giá (Tuần 4):",
         " Đánh giá chênh lệch hiệu năng trên tập Test độc lập. Triển khai API Ensemble Soft-Voting trên Flask server và kiểm thử độ trễ phản hồi.")
    ]
    
    for title, desc in timeline:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        
        rt = p.add_run(title + "\n")
        style_text_run(rt, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
        
        rd = p.add_run(desc)
        style_text_run(rd, font_name="Arial", size_pt=10, color_rgb=CHARCOAL)
        
    add_callout_box(
        doc,
        "Thông qua việc triển khai các phương pháp trên, hệ thống cam kết đạt các chỉ số đầu ra sau:\n"
        "  - Độ chính xác tổng thể (Accuracy) đạt tối thiểu 95.0% trên tập Test.\n"
        "  - Recall của lớp Taxi và Minibus cải thiện tăng ít nhất 12% so với phiên bản cũ.\n"
        "  - Độ trễ phản hồi của luồng API suy luận tích hợp Ensemble duy trì dưới 100ms trên CPU.",
        title="KPIs CAM KẾT CHẤT LƯỢNG ĐẦU RA"
    )
    
    # Save document
    output_dir = r"d:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\docs"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "DE_XUAT_PHUONG_PHAP_CAI_TIEN_FINAL.docx")
    doc.save(filepath)
    print(f"Report updated successfully at {filepath}")

if __name__ == "__main__":
    create_report()
