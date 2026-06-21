from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Bao_Cao_Cau_Hinh_Train_V3_Dong_Bo_3_Model.docx"


BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
HEADER_FILL = "F2F4F7"
CALLOUT_FILL = "F4F6F9"
BORDER = "D9E2F3"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_in: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_in * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_table_borders(table, color: str = "B7C9E2") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_table_width(table, width_in: float = 6.5) -> None:
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(int(width_in * 1440)))
    tbl_w.set(qn("w:type"), "dxa")


def set_cell_margins(table, top=80, start=120, bottom=80, end=120) -> None:
    tbl_pr = table._tbl.tblPr
    tbl_cell_mar = tbl_pr.find(qn("w:tblCellMar"))
    if tbl_cell_mar is None:
        tbl_cell_mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(tbl_cell_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tbl_cell_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tbl_cell_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def style_run(run, bold=False, color=None, size=None) -> None:
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if size:
        run.font.size = Pt(size)


def add_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_paragraph(style=f"Heading {level}")
    r = p.add_run(text)
    style_run(r, bold=True, color=BLUE if level <= 2 else DARK_BLUE)
    return p


def add_body(doc: Document, text: str):
    p = doc.add_paragraph(style="Body Text")
    p.add_run(text)
    return p


def add_bullet(doc: Document, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.05)
    p.add_run(text)
    return p


def add_callout(doc: Document, title: str, body: str):
    table = doc.add_table(rows=1, cols=1)
    set_table_width(table)
    set_table_borders(table, color=BORDER)
    set_cell_margins(table, top=120, bottom=120, start=160, end=160)
    cell = table.cell(0, 0)
    set_cell_shading(cell, CALLOUT_FILL)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    style_run(r, bold=True, color=DARK_BLUE)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.add_run(body)
    doc.add_paragraph()


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_width(table)
    set_table_borders(table)
    set_cell_margins(table)
    hdr = table.rows[0].cells
    for i, text in enumerate(headers):
        set_cell_width(hdr[i], widths[i])
        set_cell_shading(hdr[i], HEADER_FILL)
        hdr[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        style_run(r, bold=True, color=DARK_BLUE, size=9)
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            set_cell_width(cells[i], widths[i])
            cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i != 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(text)
            style_run(r, size=9)
    doc.add_paragraph()
    return table


def configure_doc(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.font.size = Pt(11)

    for name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)

    body = styles["Body Text"]
    body.font.name = "Calibri"
    body._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    body.font.size = Pt(11)
    body.paragraph_format.space_after = Pt(6)
    body.paragraph_format.line_spacing = 1.10


def main() -> None:
    doc = Document()
    configure_doc(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(3)
    r = title.add_run("Cấu Hình Train V3 Đồng Bộ Cho Ba Mô Hình")
    style_run(r, bold=True, color=DARK_BLUE, size=22)

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    r = subtitle.add_run(
        "Vehicle Type Recognition | ResNet50, ViT-B/16, YOLOv8m-cls | Augmented V2 Cleaning"
    )
    style_run(r, color="555555", size=10)

    add_callout(
        doc,
        "Kết luận cấu hình V3",
        "Ba mô hình được đồng bộ ở dữ liệu, split, input size 224, epochs 30, patience 7, "
        "CB-Focal Loss và tắt online MixUp/CutMix. YOLO truyền seed 42; ResNet/ViT cần bổ sung seed 42 trong code trước khi khẳng định tái lập hoàn toàn. Pairwise Margin Loss car/truck được thêm như "
        "loss phụ nhỏ để xử lý nhầm lẫn chính trên confusion matrix.",
    )

    add_heading(doc, "1. Bảng Cấu Hình Train V3", 1)
    add_table(
        doc,
        ["Nhóm", "ResNet50 V3", "ViT-B/16 V3", "YOLOv8m-cls V3", "Đồng bộ / Ghi chú"],
        [
            ["Dataset", "Augmented V2 Cleaning", "Augmented V2 Cleaning", "Augmented V2 Cleaning", "Đồng bộ bắt buộc"],
            ["Split", "train / valid_unseen / test", "train / valid_unseen / test", "train / valid_unseen / test", "valid_unseen chọn best; test chỉ dùng sau cùng"],
            ["Input", "224x224", "224x224", "224x224", "Đồng bộ tiền xử lý"],
            ["Backbone", "ResNet50", "ViT-B/16", "YOLOv8m-cls", "Không đồng bộ vì đây là đối tượng so sánh"],
            ["Pretrained", "ImageNet", "ImageNet", "Ultralytics pretrained", "Đều là transfer learning"],
            ["Cải tiến kiến trúc", "Multi-scale feature fusion", "Multi-layer CLS fusion", "YOLOv8m backbone", "Khác theo kiến trúc"],
            ["Epochs / patience", "30 / 7", "30 / 7", "30 / 7", "Đồng bộ"],
            ["Batch", "32", "32", "32", "Đồng bộ; giảm riêng khi GPU không đủ bộ nhớ và phải ghi rõ"],
            ["Optimizer", "AdamW", "AdamW", "AdamW", "Đồng bộ ở loại optimizer"],
            ["LR", "head 1e-3; backbone 1e-5", "head 1e-3; backbone 1e-5", "lr0 1.5e-4", "Không ép cùng số vì engine khác"],
            ["Scheduler", "ReduceLROnPlateau", "ReduceLROnPlateau", "Ultralytics schedule", "Khác hợp lý theo framework"],
            ["Freeze strategy", "5 epoch head_only, sau đó partial", "5 epoch head_only, sau đó partial", "Không tách phase", "Khác do YOLO engine"],
            ["Loss chính", "CB-Focal", "CB-Focal", "CB-Focal", "Đồng bộ"],
            ["Loss phụ", "Pair-margin car/truck", "Pair-margin car/truck", "Pair-margin car/truck", "Đồng bộ V3"],
            ["cb_beta / cb_gamma", "0.999 / 2.0", "0.999 / 2.0", "0.999 / 2.0", "Đồng bộ"],
            ["Label smoothing", "0.1", "0.1", "0.05 (mặc định engine)", "YOLO giữ 0.05 để tránh làm mềm ranh giới car/truck"],
            ["MixUp/CutMix", "0.0 / 0.0", "0.0 / 0.0", "mixup 0.0", "Đồng bộ tắt online mixup"],
            ["Seed", "Chưa truyền trong command", "Chưa truyền trong command", "42", "Cần bổ sung seed 42 cho ResNet/ViT để đồng bộ tái lập"],
        ],
        [0.85, 1.25, 1.25, 1.25, 1.9],
    )

    add_heading(doc, "2. Cấu Hình Nên Đồng Bộ Và Lý Do", 1)
    add_table(
        doc,
        ["Thành phần", "Giá trị V3", "Lý do chọn"],
        [
            ["Dataset / split", "Augmented V2 Cleaning; train-valid_unseen-test", "Đảm bảo ba mô hình học và đánh giá trên cùng phân phối dữ liệu."],
            ["Input size", "224x224", "Giữ tiền xử lý công bằng; không tăng size riêng cho YOLO."],
            ["Epochs / patience", "30 / 7", "Cùng giới hạn train và early stopping."],
            ["Seed", "YOLO: 42; ResNet/ViT: cần bổ sung", "Chỉ được xem là đồng bộ tái lập khi cả ba code cùng đặt seed 42."],
            ["CB-Focal", "beta 0.999; gamma 2.0", "Xử lý mất cân bằng số ảnh gốc và tập trung mẫu khó."],
            ["Pair-margin", "car/truck; margin 0.35; lambda 0.15", "Tách rõ logit car/truck; chỉ áp dụng cho cặp nhầm chính, không tăng class weight toàn cục."],
            ["Online MixUp/CutMix", "Tắt", "Tránh làm mờ ranh giới taxi/car và truck/minibus."],
            ["Test protocol", "Không dùng test khi train", "Test chỉ dùng để báo cáo cuối, tránh chọn model theo test."],
        ],
        [1.65, 1.65, 3.2],
    )

    add_heading(doc, "3. Vì Sao Chọn CB-Focal + Pairwise Margin", 1)
    add_body(
        doc,
        "CB-Focal được dùng làm loss nền vì dataset train sau offline augmentation cân bằng về số file, "
        "nhưng số ảnh gốc giữa các lớp vẫn lệch. Thành phần Class-Balanced bù cho lớp ít ảnh gốc; "
        "thành phần Focal tập trung vào mẫu khó."
    )
    add_body(
        doc,
        "Pairwise Margin Loss được thêm với trọng số nhỏ để xử lý đúng lỗi còn nổi bật trên ma trận nhầm lẫn: "
        "car bị dự đoán thành truck và truck bị dự đoán thành car. Loss này chỉ phạt khi logit của lớp đúng "
        "không vượt đủ xa logit của lớp dễ nhầm, do đó ít ảnh hưởng hơn so với tăng class weight toàn cục."
    )
    add_table(
        doc,
        ["Lựa chọn loss", "Ưu điểm", "Hạn chế", "Quyết định V3"],
        [
            ["CrossEntropy", "Đơn giản, ổn định", "Không xử lý lớp khó/lệch ảnh gốc", "Không chọn làm chính"],
            ["Weighted CE", "Cân bằng lớp", "Không tập trung mẫu khó", "Yếu hơn CB-Focal"],
            ["Focal Loss", "Tập trung mẫu khó", "Không bù số ảnh gốc", "Kết hợp trong CB-Focal"],
            ["CB-Focal", "Xử lý lớp ít ảnh gốc và mẫu khó", "Cần tune beta/gamma", "Loss chính"],
            ["SupCon/Triplet", "Mạnh về embedding", "Cần sửa pipeline sâu, batch/mining phức tạp", "Chưa dùng V3"],
            ["Pair-margin", "Nhắm đúng car/truck", "Cần lambda nhỏ để tránh lệch lớp khác", "Loss phụ"],
        ],
        [1.35, 1.7, 1.85, 1.6],
    )

    add_heading(doc, "4. Những Điểm Khác Biệt Được Chấp Nhận", 1)
    add_table(
        doc,
        ["Khác biệt", "Vì sao chấp nhận", "Cách trình bày trong báo cáo"],
        [
            ["Backbone và feature fusion", "Đây là đối tượng so sánh: ResNet, ViT và YOLO có cơ chế biểu diễn khác nhau.", "Khác kiến trúc, cùng dữ liệu và protocol."],
            ["LR và scheduler", "LR tuyệt đối không tương đương giữa custom PyTorch và Ultralytics engine.", "Đồng bộ optimizer AdamW, không ép cùng LR."],
            ["Freeze/unfreeze", "ResNet/ViT dễ tách head/backbone; YOLO engine không tương tự trực tiếp.", "Khác do framework."],
            ["YOLO label smoothing 0.05", "Smoothing 0.1 có thể làm mềm ranh giới car/truck/taxi.", "Giữ như regularization đặc thù YOLO, hoặc thử ablation nếu cần."],
        ],
        [1.55, 2.6, 2.35],
    )

    add_heading(doc, "5. Command Cấu Hình V3 Tóm Tắt", 1)
    add_table(
        doc,
        ["Model", "Các tham số bắt buộc trong command"],
        [
            ["ResNet50", "--epochs 30 --patience 7 --batch_size 32 --cb_beta 0.999 --cb_gamma 2.0 --label_smoothing 0.1 --mixup_alpha 0.0 --cutmix_alpha 0.0 --pair_margin_classes car,truck --pair_margin 0.35 --pair_lambda 0.15"],
            ["ViT-B/16", "--epochs 30 --patience 7 --batch_size 32 --cb_beta 0.999 --cb_gamma 2.0 --label_smoothing 0.1 --mixup_alpha 0.0 --cutmix_alpha 0.0 --pair_margin_classes car,truck --pair_margin 0.35 --pair_lambda 0.15"],
            ["YOLOv8m-cls", "--epochs 30 --patience 7 --imgsz 224 --seed 42 --cb_beta 0.999 --cb_gamma 2.0 --lr0 0.00015 --dropout 0.15 --mixup 0.0 --pair_margin_classes car,truck --pair_margin 0.35 --pair_lambda 0.15"],
        ],
        [1.35, 5.15],
    )

    add_heading(doc, "6. Kết Luận Báo Cáo", 1)
    add_bullet(doc, "V3 đồng bộ dataset, split, input size, epoch, patience, loss nền và tắt online mixup. Seed 42 hiện đã có ở YOLO; cần bổ sung cho ResNet/ViT để tái lập hoàn toàn.")
    add_bullet(doc, "Các khác biệt còn lại thuộc về bản chất kiến trúc hoặc engine huấn luyện, không nên ép giống tuyệt đối.")
    add_bullet(doc, "Pairwise Margin Loss là cải tiến có mục tiêu, dùng để giảm nhầm lẫn car/truck đã quan sát trên confusion matrix.")
    add_bullet(doc, "Kết quả cuối cần báo cáo cả accuracy, macro F1, weighted F1 và top confusions; không chỉ dựa vào accuracy.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
