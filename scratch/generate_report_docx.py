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

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
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
    """Applies clean borders to a table."""
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
    
    # Custom left border (blue/teal) and remove others
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
    run_text.font.size = Pt(10.5)
    run_text.font.italic = True
    run_text.font.color.rgb = RGBColor(80, 80, 80)
    
    # Add an empty paragraph after the callout for spacing
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
    
    # Page Margins (Normal: 1 inch on all sides)
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
        hrun = hp.add_run("Báo cáo Đề xuất Cải tiến Hệ thống Nhận diện Phương tiện | Dự án Nhóm")
        style_text_run(hrun, font_name="Arial", size_pt=8.5, color_rgb=RGBColor(120, 120, 120))
        
        # Footer (Page 2 onwards)
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Trang 2 / 5 (Tài liệu lưu hành nội bộ)")
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
    t_run = p_title.add_run("BẢN ĐỀ XUẤT GIẢI PHÁP CHUYÊN NGHIỆP\nNÂNG CAO ĐỘ CHÍNH XÁC HỆ THỐNG NHẬN DIỆN PHƯƠNG TIỆN")
    style_text_run(t_run, font_name="Arial", size_pt=20, bold=True, color_rgb=NAVY)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(24)
    s_run = p_sub.add_run("Phân tích chi tiết sai số từ thực trạng 3 mô hình (YOLOv8-cls, ResNet-50, ViT-Base) và xây dựng lộ trình cải tiến tối ưu hóa xử lý ảnh số")
    style_text_run(s_run, font_name="Arial", size_pt=12, italic=True, color_rgb=TEAL)
    
    # Horizontal separator line
    p_sep = doc.add_paragraph()
    p_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sep.paragraph_format.space_after = Pt(24)
    sep_run = p_sep.add_run("—" * 50)
    style_text_run(sep_run, font_name="Arial", size_pt=10, color_rgb=MUTED)
    
    # ------------------ INTRODUCTION ------------------
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.line_spacing = 1.15
    p_intro.paragraph_format.space_after = Pt(12)
    irun = p_intro.add_run(
        "Báo cáo này được xây dựng nhằm mục đích nghiên cứu chuyên sâu các lỗi phân loại hệ thống "
        "của 3 kiến trúc mô hình tiêu biểu trong dự án nhận diện phương tiện giao thông (Vehicle Type Recognition): "
        "YOLOv8-cls (đại diện cho dòng lightweight/real-time), ResNet-50 (mô hình baseline CNN tiêu chuẩn) và "
        "Vision Transformer - ViT-Base (mô hình học ngữ cảnh toàn cục). Thông qua việc đối chiếu và phân tích "
        "ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử độc lập (Test Split gồm 3,601 ảnh), nhóm đề xuất "
        "một hệ thống các giải pháp đồng bộ từ tiền xử lý ảnh số đến tối ưu hóa hàm mất mát và kiến trúc mạng "
        "để nâng hiệu năng tổng thể vượt ngưỡng 96%."
    )
    style_text_run(irun, font_name="Arial", size_pt=11, color_rgb=CHARCOAL)
    
    # ------------------ SECTION I ------------------
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(8)
    h1_run = h1.add_run("I. PHÂN TÍCH CHI TIẾT SỐ LIỆU HIỆN TRẠNG VÀ SAI SỐ HỆ THỐNG")
    style_text_run(h1_run, font_name="Arial", size_pt=14, bold=True, color_rgb=NAVY)
    
    p_desc1 = doc.add_paragraph()
    p_desc1.paragraph_format.space_after = Pt(10)
    p_desc1.paragraph_format.line_spacing = 1.15
    run_desc1 = p_desc1.add_run(
        "Các mô hình được huấn luyện trong điều kiện đồng bộ hóa toàn diện về cấu hình tiền xử lý cơ bản (Base Pipeline: Resize giữ tỷ lệ và Zero-padding về 224x224, chuẩn hóa mean/std theo ImageNet), "
        "bộ tối ưu hóa AdamW, tốc độ học ban đầu lr0=1e-3, warmup 5 epochs, và hệ số suy giảm trọng số decay=1e-4. "
        "Kết quả đánh giá hiệu năng tổng thể trên tập kiểm thử Test Split (3,601 ảnh) được trình bày chi tiết trong Bảng 1."
    )
    style_text_run(run_desc1, font_name="Arial", size_pt=11, color_rgb=CHARCOAL)
    
    # Table 1: Model Comparison
    t1 = doc.add_table(rows=5, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)
    
    headers_t1 = ["Chỉ số đo lường", "YOLOv8-cls (Lightweight)", "ResNet-50 (Baseline CNN)", "Vision Transformer (ViT-Base)"]
    col_widths_t1 = [Inches(2.5), Inches(1.5), Inches(1.5), Inches(1.5)]
    
    # Style Header
    for idx, name in enumerate(headers_t1):
        cell = t1.cell(0, idx)
        cell.width = col_widths_t1[idx]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
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
        # Determine background color for row (alternate shading)
        bg_color = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            cell = t1.cell(row_idx, col_idx)
            cell.width = col_widths_t1[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            
            # Bold for values of ViT (highest performing) and make it stand out
            is_vit_highest = (col_idx == 3 and row_idx in [1, 3, 4])
            style_text_run(run, font_name="Arial", size_pt=9.5, bold=is_vit_highest, 
                           color_rgb=NAVY if is_vit_highest else CHARCOAL)
            
    p_caption1 = doc.add_paragraph()
    p_caption1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_caption1.paragraph_format.space_before = Pt(6)
    p_caption1.paragraph_format.space_after = Pt(18)
    run_cap1 = p_caption1.add_run("Bảng 1: So sánh hiệu năng tổng thể của 3 mô hình trên tập kiểm thử độc lập")
    style_text_run(run_cap1, font_name="Arial", size_pt=9, italic=True, color_rgb=MUTED)
    
    # Class-level analysis
    p_desc2 = doc.add_paragraph()
    p_desc2.paragraph_format.space_after = Pt(10)
    p_desc2.paragraph_format.line_spacing = 1.15
    run_desc2 = p_desc2.add_run(
        "Để nhận diện rõ nét điểm yếu cục bộ của từng kiến trúc mô hình, Bảng 2 trình bày chi tiết chỉ số F1-Score "
        "của 10 lớp phương tiện được sắp xếp dựa trên sự phân bố số mẫu hỗ trợ thực tế (Support)."
    )
    style_text_run(run_desc2, font_name="Arial", size_pt=11, color_rgb=CHARCOAL)
    
    # Table 2: Class-wise F1
    t2 = doc.add_table(rows=11, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)
    
    headers_t2 = ["Lớp phương tiện", "Số mẫu (Support)", "YOLOv8-cls F1", "ResNet-50 F1", "ViT-Base F1"]
    col_widths_t2 = [Inches(2.2), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)]
    
    # Style Header
    for idx, name in enumerate(headers_t2):
        cell = t2.cell(0, idx)
        cell.width = col_widths_t2[idx]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
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
            
            # Highlight low F1-scores (< 85%) in dark red/orange to raise awareness
            is_low_f1 = (col_idx >= 2 and float(text.replace('%', '')) < 85.0)
            if is_low_f1:
                style_text_run(run, font_name="Arial", size_pt=9.5, bold=True, color_rgb=RGBColor(180, 40, 40))
            else:
                style_text_run(run, font_name="Arial", size_pt=9.5, color_rgb=CHARCOAL)
                
    p_caption2 = doc.add_paragraph()
    p_caption2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_caption2.paragraph_format.space_before = Pt(6)
    p_caption2.paragraph_format.space_after = Pt(18)
    run_cap2 = p_caption2.add_run("Bảng 2: Chỉ số F1-Score chi tiết của 10 lớp phương tiện trên tập kiểm thử")
    style_text_run(run_cap2, font_name="Arial", size_pt=9, italic=True, color_rgb=MUTED)
    
    # Detailed analysis of error modes
    p_err_intro = doc.add_paragraph()
    p_err_intro.paragraph_format.space_before = Pt(6)
    p_err_intro.paragraph_format.space_after = Pt(10)
    p_err_intro.paragraph_format.line_spacing = 1.15
    run_err_intro = p_err_intro.add_run(
        "Phân tích sâu ma trận nhầm lẫn chỉ ra 3 nguyên nhân cốt lõi gây suy giảm độ chính xác hệ thống:"
    )
    style_text_run(run_err_intro, font_name="Arial", size_pt=11, bold=True, color_rgb=CHARCOAL)
    
    # 3 Error bullet points
    errors = [
        ("1. Điểm nghẽn phân loại Taxi ↔ Car (Bỏ sót đặc trưng cục bộ):", 
         " Lớp taxi (91 mẫu) có chỉ số Recall vô cùng thấp. YOLOv8-cls bỏ sót 33/91 ảnh taxi (36.3% nhầm thành car), ResNet-50 nhầm 24/91 ảnh, và ViT-Base nhầm 13/91 ảnh. "
         "Nguyên nhân vật lý là kiểu dáng hình khối của taxi và xe con thông thường hoàn toàn đồng dạng. Sự khác biệt chỉ nằm ở mào taxi trên nóc hoặc tem chữ dán bên cửa xe (chiếm dưới 2% diện tích ảnh). "
         "Khi resize ảnh trực tiếp về 224x224, các đặc trưng tần số cao (high-frequency) này bị triệt tiêu hoàn toàn sau các lớp tích chập downsampling."),
         
        ("2. Điểm nghẽn phân loại Minibus ↔ Car/Truck (Tính mập mờ hình học):",
         " Lớp minibus có F1-Score rất thấp (79.9% ở YOLO, 81.3% ở ResNet, 80.6% ở ViT) và bị nhầm lẫn lưỡng cực: nhầm sang Car (dòng MPV/SUV dài) và Truck (xe tải nhỏ). "
         "Về mặt vật lý, xe khách 16 chỗ (Minibus) có tỷ lệ khung hình (Aspect Ratio) hình hộp dài lấp lửng giữa ô tô con lớn và xe tải nhỏ. "
         "Quy trình resize và padding thông thường nếu làm thay đổi tỉ lệ hoặc góc quan sát xiên sẽ khiến ranh giới quyết định (decision boundary) của mô hình bị chồng lấn."),
         
        ("3. Điểm nghẽn Truck ↔ Car & Bus (Nhiễu thời tiết và thiếu tương phản):",
         " Recall của truck ở YOLOv8-cls chỉ đạt 80.1% (nhầm 37 ảnh sang car, 15 sang bus). "
         "Nguyên nhân chủ yếu do các ảnh chụp xe tải ban đêm (night) hoặc trời mưa (rain) bị mất độ tương phản biên dạng thùng xe vuông vức, "
         "khiến mô hình nhầm lẫn góc vuông thùng xe thành cabin bo tròn của xe buýt hoặc ô tô con.")
    ]
    
    for title, desc in errors:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        
        r_title = p.add_run(title)
        style_text_run(r_title, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
        
        r_desc = p.add_run(desc)
        style_text_run(r_desc, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
        
    add_callout_box(
        doc,
        "Hiện tượng nhầm lẫn giữa các lớp có độ tương đồng thị giác cao (Fine-grained classification) như Taxi vs Car "
        "và sự mất cân bằng nghiêm trọng về mặt mẫu hỗ trợ (Taxi 91 mẫu vs Car 854 mẫu) chính là rào cản lớn nhất "
        "ngăn chặn hệ thống đạt được hiệu năng tối ưu. Việc điều chỉnh mô hình đơn thuần không giải quyết được vấn đề "
        "mà bắt buộc phải áp dụng các kỹ thuật xử lý ảnh số chuyên sâu.",
        title="NHẬN ĐỊNH CỦA CHUYÊN GIA"
    )
    
    # ------------------ SECTION II ------------------
    h2 = doc.add_heading(level=1)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(8)
    h2_run = h2.add_run("II. BẢNG ĐỀ XUẤT PHƯƠNG PHÁP CẢI TIẾN CHUYÊN NGHIỆP")
    style_text_run(h2_run, font_name="Arial", size_pt=14, bold=True, color_rgb=NAVY)
    
    p_desc3 = doc.add_paragraph()
    p_desc3.paragraph_format.space_after = Pt(12)
    p_desc3.paragraph_format.line_spacing = 1.15
    run_desc3 = p_desc3.add_run(
        "Nhằm khắc phục triệt để các sai số nêu trên, nhóm đề xuất ma trận giải pháp kỹ thuật tích hợp "
        "giữa Xử lý ảnh số truyền thống và tối ưu hóa Mô hình Học sâu (Deep Learning Optimization):"
    )
    style_text_run(run_desc3, font_name="Arial", size_pt=11, color_rgb=CHARCOAL)
    
    # Large improvements table (6 columns/rows)
    t3 = doc.add_table(rows=7, cols=5)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3, color="B0C4DE", sz="6") # Light steel blue borders
    
    headers_t3 = ["STT", "Phương Pháp Đề Xuất", "Giải Pháp Chi Tiết & Kỹ Thuật", "Dẫn Chứng Số Liệu Hiện Tại Cần Khắc Phục", "Cơ Sở Khoa Học & Cơ Chế Hoạt Động"]
    col_widths_t3 = [Inches(0.4), Inches(1.5), Inches(1.9), Inches(1.7), Inches(2.0)]
    
    # Style Header
    for idx, name in enumerate(headers_t3):
        cell = t3.cell(0, idx)
        cell.width = col_widths_t3[idx]
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=140, bottom=140, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(name)
        style_text_run(run, font_name="Arial", size_pt=9, bold=True, color_rgb=RGBColor(255, 255, 255))
        
    data_t3 = [
        [
            "1",
            "Reflective Padding\n(Đệm biên đối xứng)",
            "Thay thế hàm zero_pad_to_square (đệm viền đen) bằng kỹ thuật lấy đối xứng biên ảnh qua gương (Mirror/Reflective padding).",
            "Biên ảnh chuyển đột ngột từ ảnh thực sang màu đen tạo ra các cạnh sắc nhọn (High-frequency edges) làm nhiễu bộ lọc tích chập.",
            "Triệt tiêu xung nhiễu tần số cao tại biên. Giúp phân phối giá trị trung bình (Mean) của các activation map ổn định hơn, hướng sự tập trung (Attention) vào trung tâm chứa phương tiện."
        ],
        [
            "2",
            "CLAHE trong không gian màu LAB\n(Cân bằng sáng cục bộ)",
            "Chuyển ảnh sang không gian màu LAB, áp dụng CLAHE trên kênh L (Luminance) với ngưỡng giới hạn tương phản (clip limit = 2.0).",
            "Dữ liệu thời tiết ngoại tuyến (night chiếm 10%, rain chiếm 10%) làm mờ biên dạng thùng xe của Truck và Bus, gây nhầm lẫn chéo 62 mẫu trên YOLO.",
            "Tăng cường độ tương phản cục bộ mà không gây cháy sáng hay nhiễu hạt (noise amplification) như Histogram Equalization thông thường. Giúp làm nổi rõ biên của xe tải trong bóng tối."
        ],
        [
            "3",
            "Tăng Độ Phân Giải Đầu Vào (384x384) & Multi-Scale Training",
            "Tăng độ phân giải từ 224x224 lên 384x384 pixel. Áp dụng kỹ thuật huấn luyện đa tỷ lệ (MST) ngẫu nhiên mỗi 10 epochs.",
            "Recall của Taxi cực thấp (63.7% ở YOLO) do mào xe bị mờ hóa khi resize quá nhỏ. Xe Minibus bị nhầm sang Car (22 mẫu trên ViT) do mất chi tiết cửa kính dài.",
            "Định lý lấy mẫu Nyquist-Shannon: Tăng độ phân giải giúp giữ lại các tần số không gian cao. MST tạo ra sự bất biến về kích thước (Scale Invariance), giúp mạng nhận diện được mào xe taxi ở cả cự ly xa lẫn gần."
        ],
        [
            "4",
            "Focal Loss kết hợp Class-Balanced Weighting",
            "Thay thế Cross-Entropy Loss bằng Focal Loss kết hợp trọng số nghịch đảo mẫu hiệu dụng (Effective Number of Samples Weighting).",
            "Lớp taxi chỉ có 91 mẫu, lớp helicopter có 67 mẫu (thiểu số) so với boat (890 mẫu) và car (854 mẫu) dẫn đến mô hình bị thiên lệch (bias) về lớp đa số.",
            "Focal Loss tự động hạ trọng số của các mẫu dễ (Boat, Motorcycle) và nhân dòng lỗi của các mẫu khó phân loại sai lên gamma lần. Class-Balanced Weighting giúp định hình lại không gian vector tối ưu (Loss Space) cân bằng hơn cho lớp thiểu số."
        ],
        [
            "5",
            "Augmentation hướng đối tượng (ROI-Focused Random Erasing)",
            "Thực hiện che khuất ngẫu nhiên (Random Erasing) có giới hạn tọa độ (chỉ áp dụng ở nửa trên cabin xe con và xe tải).",
            "33 Taxi bị nhầm thành Car và 37 Truck bị nhầm thành Car do mô hình chỉ học các đặc trưng dễ ăn điểm như bánh xe, đầu xe.",
            "Buộc mô hình phải học các đặc trưng phân tán (Distributed Representations). Khi phần nóc xe hoặc cabin bị che khuất, mô hình buộc phải tìm kiếm các đặc trưng khác (đèn pha, lưới tản nhiệt, đuôi xe) để ra quyết định, giảm Overfitting."
        ],
        [
            "6",
            "Ensemble Soft-Voting (ViT + ResNet-50 + YOLOv8)",
            "Kết hợp phân phối xác suất (Soft Probability) của 3 mô hình theo trọng số:\nP = 0.5*P(ViT) + 0.3*P(ResNet50) + 0.2*P(YOLO)",
            "ViT đạt Accuracy cao nhất (94.06%) nhưng inference chậm. YOLO chạy rất nhanh nhưng Recall Taxi yếu (63.7%).",
            "Kết hợp sức mạnh của 2 trường phái kiến trúc: Global Attention (ViT học bối cảnh toàn cục rất tốt) và Local Inductive Bias (ResNet/YOLO học chi tiết biên cạnh cục bộ rất tốt). Lỗi sai của mạng này sẽ được bù đắp bởi mạng khác."
        ]
    ]
    
    for row_idx, row_data in enumerate(data_t3, start=1):
        bg_color = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(row_data):
            cell = t3.cell(row_idx, col_idx)
            cell.width = col_widths_t3[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
            
            # Vertical alignment: Center
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            
            # Align center for STT and Method, left for details
            if col_idx in [0]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif col_idx in [1]:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            run = p.add_run(text)
            
            # Format STT and Method title as bold
            is_bold = (col_idx in [0, 1])
            style_text_run(run, font_name="Arial", size_pt=9.5, bold=is_bold, color_rgb=NAVY if is_bold else CHARCOAL)
            
    p_caption3 = doc.add_paragraph()
    p_caption3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_caption3.paragraph_format.space_before = Pt(6)
    p_caption3.paragraph_format.space_after = Pt(18)
    run_cap3 = p_caption3.add_run("Bảng 3: Ma trận đề xuất giải pháp cải tiến chuyên nghiệp tích hợp xử lý ảnh và học sâu")
    style_text_run(run_cap3, font_name="Arial", size_pt=9, italic=True, color_rgb=MUTED)
    
    # ------------------ SECTION III ------------------
    h3 = doc.add_heading(level=1)
    h3.paragraph_format.space_before = Pt(18)
    h3.paragraph_format.space_after = Pt(8)
    h3_run = h3.add_run("III. CƠ SỞ TOÁN HỌC CHO PHƯƠNG PHÁP TỐI ƯU HÀM MẤT MÁT")
    style_text_run(h3_run, font_name="Arial", size_pt=14, bold=True, color_rgb=NAVY)
    
    p_math_intro = doc.add_paragraph()
    p_math_intro.paragraph_format.space_after = Pt(10)
    p_math_intro.paragraph_format.line_spacing = 1.15
    run_math_intro = p_math_intro.add_run(
        "Để giải quyết triệt để lỗi phân loại sai trên nhóm lớp thiểu số (Taxi, Minibus), "
        "nhóm đề xuất cấu hình hàm mất mát mới tích hợp Class-Balanced Loss và Focal Loss trong chu kỳ huấn luyện tiếp theo:"
    )
    style_text_run(run_math_intro, font_name="Arial", size_pt=11, color_rgb=CHARCOAL)
    
    # Formula 1
    p_f1 = doc.add_paragraph()
    p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_f1.paragraph_format.space_before = Pt(8)
    p_f1.paragraph_format.space_after = Pt(8)
    run_f1 = p_f1.add_run("1. Trọng số lớp hiệu dụng (Class-Balanced Weight):\n")
    style_text_run(run_f1, font_name="Arial", size_pt=11, bold=True, color_rgb=TEAL)
    
    run_eq1 = p_f1.add_run("W_i = (1 - \u03b2) / (1 - \u03b2^{n_i})")
    style_text_run(run_eq1, font_name="Courier New", size_pt=12, bold=True, color_rgb=NAVY)
    
    p_f1_desc = doc.add_paragraph()
    p_f1_desc.paragraph_format.left_indent = Inches(0.25)
    p_f1_desc.paragraph_format.space_after = Pt(10)
    p_f1_desc.paragraph_format.line_spacing = 1.15
    run_f1_desc = p_f1_desc.add_run(
        "Trong đó: \u03b2 = 0.999 (hệ số điều chỉnh mức độ giao thoa không gian đặc trưng), "
        "n_i là số lượng mẫu thực tế của lớp i trong tập huấn luyện. Trọng số này tỉ lệ nghịch với thể tích không gian "
        "bao phủ hiệu dụng của các mẫu, giúp cân bằng lại lực kéo gradient giữa lớp đa số (Boat, Car) và lớp thiểu số (Taxi, Helicopter)."
    )
    style_text_run(run_f1_desc, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # Formula 2
    p_f2 = doc.add_paragraph()
    p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_f2.paragraph_format.space_before = Pt(8)
    p_f2.paragraph_format.space_after = Pt(8)
    run_f2 = p_f2.add_run("2. Hàm mất mát Focal Loss sửa đổi (Weighted Focal Loss):\n")
    style_text_run(run_f2, font_name="Arial", size_pt=11, bold=True, color_rgb=TEAL)
    
    run_eq2 = p_f2.add_run("L_{CB-Focal}(p_t) = - W_i \u22c5 (1 - p_t)^\u03b3 \u22c5 log(p_t)")
    style_text_run(run_eq2, font_name="Courier New", size_pt=12, bold=True, color_rgb=NAVY)
    
    p_f2_desc = doc.add_paragraph()
    p_f2_desc.paragraph_format.left_indent = Inches(0.25)
    p_f2_desc.paragraph_format.space_after = Pt(12)
    p_f2_desc.paragraph_format.line_spacing = 1.15
    run_f2_desc = p_f2_desc.add_run(
        "Trong đó: p_t là xác suất dự đoán của lớp chính xác, \u03b3 = 2.0 (hệ số tập trung hạt nhân). "
        "Cơ chế hoạt động: Khi mô hình phân loại sai một mẫu Taxi khó (p_t thấp, ví dụ 0.2), đại lượng điều chế (1 - 0.2)^2 = 0.64 sẽ giữ lại phần lớn giá trị lỗi để ép mô hình học tiếp. "
        "Ngược lại, nếu mô hình nhận diện đúng một mẫu Car dễ (p_t = 0.95), đại lượng điều chế (1 - 0.95)^2 = 0.0025 sẽ triệt tiêu gần như toàn bộ giá trị loss của mẫu đó. "
        "Nhờ vậy, mô hình sẽ không bị bão hòa bởi lớp đa số và tập trung tối ưu hóa các ca khó phân loại."
    )
    style_text_run(run_f2_desc, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
    
    # ------------------ SECTION IV ------------------
    h4 = doc.add_heading(level=1)
    h4.paragraph_format.space_before = Pt(18)
    h4.paragraph_format.space_after = Pt(8)
    h4_run = h4.add_run("IV. LỘ TRÌNH TRIỂN KHAI VÀ CAM KẾT CHẤT LƯỢNG")
    style_text_run(h4_run, font_name="Arial", size_pt=14, bold=True, color_rgb=NAVY)
    
    p_desc4 = doc.add_paragraph()
    p_desc4.paragraph_format.space_after = Pt(12)
    p_desc4.paragraph_format.line_spacing = 1.15
    run_desc4 = p_desc4.add_run(
        "Để đảm bảo tính khả thi và đo lường được hiệu quả của đề xuất cải tiến, "
        "lộ trình triển khai được chia làm 3 giai đoạn cụ thể:"
    )
    style_text_run(run_desc4, font_name="Arial", size_pt=11, color_rgb=CHARCOAL)
    
    # Timeline bullets
    timeline = [
        ("Giai đoạn 1: Nâng cấp Pipeline Tiền xử lý (Tuần 1-2):",
         " Thay đổi thuật toán padding từ Zero Padding sang Reflective Padding trên toàn hệ thống (Train/Valid/Test). Tích hợp module chuyển đổi LAB và áp dụng CLAHE cho tập Train nhiễu tối và mưa. Thực hiện đo đạc biến thiên phổ tần số biên của ảnh đầu vào."),
        ("Giai đoạn 2: Tái cấu trúc cấu hình Huấn luyện & Hàm Loss (Tuần 3-4):",
         " Tăng độ phân giải đầu vào lên 384x384. Triển khai Class-Balanced Focal Loss trên Kaggle notebook. Cấu hình ROI-focused Random Erasing cho các batch huấn luyện. Thực hiện train song song 3 mô hình."),
        ("Giai đoạn 3: Tích hợp Ensemble & Kiểm thử Độc lập (Tuần 5):",
         " Xây dựng API tích hợp Soft-Voting Ensemble trên Flask. Chạy đánh giá chính thức trên tập Test độc lập và đối chiếu chênh lệch delta hiệu năng so với phiên bản cũ.")
    ]
    
    for stage_title, stage_desc in timeline:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        
        r_title = p.add_run(stage_title)
        style_text_run(r_title, font_name="Arial", size_pt=10.5, bold=True, color_rgb=TEAL)
        
        r_desc = p.add_run(stage_desc)
        style_text_run(r_desc, font_name="Arial", size_pt=10.5, color_rgb=CHARCOAL)
        
    add_callout_box(
        doc,
        "Với việc áp dụng đồng bộ các giải pháp cải tiến nêu trên, chúng tôi cam kết sẽ:\n"
        "1. Nâng cao độ chính xác tổng thể (Accuracy) của hệ thống lên mức tối thiểu 96.0%.\n"
        "2. Đưa Recall của nhóm lớp khó (Taxi và Minibus) đạt trên 88.0%.\n"
        "3. Tối ưu hóa API Flask đảm bảo độ trễ suy luận của mô hình Ensemble dưới 100ms thông qua ONNX Quantization.",
        title="CAM KẾT CHẤT LƯỢNG ĐẦU RA (KPIs)"
    )
    
    # Save document
    output_dir = r"d:\Xu Ly Anh So\ProjectNhom\vehicle-type-recognition\docs"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "DE_XUAT_PHUONG_PHAP_CAI_TIEN.docx")
    doc.save(filepath)
    print(f"Report saved successfully to {filepath}")

if __name__ == "__main__":
    create_report()
