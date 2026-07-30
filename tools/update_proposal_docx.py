from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "De_cuong_DASP-NIDS_Thuc_nghiem.backup_20260722.docx"
OUTPUT = ROOT / "De_cuong_DASP-NIDS_Thuc_nghiem.docx"
FIGURES = ROOT / "ACCESS_latex_template_20260513" / "figures"


def replace_text_keep_first_run(paragraph, text: str) -> None:
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def remove_body_from(document: Document, first_heading_text: str) -> None:
    body = document._element.body
    start = None
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == first_heading_text:
            start = paragraph._element
            break
    if start is None:
        raise RuntimeError(f"Không tìm thấy mốc nội dung: {first_heading_text}")
    remove = False
    for child in list(body):
        if child is start:
            remove = True
        if remove and child.tag != qn("w:sectPr"):
            body.remove(child)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=90, bottom=90, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    header = table.rows[0]
    repeat_table_header(header)
    for index, text in enumerate(headers):
        cell = header.cells[index]
        cell.text = text
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "D9EAF7")
        set_cell_margins(cell)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                run.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
    for row_data in rows:
        row = table.add_row()
        prevent_row_split(row)
        for index, text in enumerate(row_data):
            cell = row.cells[index]
            cell.text = str(text)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9.5)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_heading(document: Document, text: str, level: int = 1):
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.widow_control = True
    return paragraph


def add_body(document: Document, text: str, bold_prefix: str | None = None):
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.widow_control = True
    if bold_prefix and text.startswith(bold_prefix):
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        paragraph.add_run(text[len(bold_prefix):])
    else:
        paragraph.add_run(text)
    return paragraph


def add_bullet(document: Document, text: str):
    paragraph = document.add_paragraph(text, style="List Bullet")
    paragraph.paragraph_format.widow_control = True
    return paragraph


def add_number(document: Document, text: str):
    paragraph = document.add_paragraph(text, style="List Number")
    paragraph.paragraph_format.widow_control = True
    return paragraph


def add_equation(document: Document, text: str):
    paragraph = document.add_paragraph(style="EquationText")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run(text)
    return paragraph


def add_caption(document: Document, text: str):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = False
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    return paragraph


def add_figure(document: Document, filename: str, caption: str, width: float = 6.25):
    path = FIGURES / filename
    if not path.exists():
        raise FileNotFoundError(path)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    add_caption(document, caption)


def add_page_break(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.add_run().add_break()
    paragraph.runs[0].add_break()


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11.5)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    for name, size, color in (
        ("Heading 1", 14, "1F4E79"),
        ("Heading 2", 12, "2F5597"),
    ):
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True
    if "EquationText" in document.styles:
        equation = document.styles["EquationText"]
        equation.font.name = "Cambria Math"
        equation.font.size = Pt(10.5)
        equation.paragraph_format.space_before = Pt(3)
        equation.paragraph_format.space_after = Pt(6)
    for name in ("List Bullet", "List Number"):
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(11.5)
        style.paragraph_format.space_after = Pt(3)


def main() -> None:
    document = Document(SOURCE)
    configure_styles(document)

    cover_paragraphs = document.paragraphs
    replace_text_keep_first_run(cover_paragraphs[2], "DASP-v5: Guarded Utility-Portfolio Adaptation for Online Neural Network Intrusion Detection Under Concept Drift")
    replace_text_keep_first_run(cover_paragraphs[4], "Thích nghi theo danh mục utility có cơ chế bảo vệ cho hệ thống phát hiện xâm nhập mạng nơ-ron trực tuyến dưới concept drift")
    replace_text_keep_first_run(cover_paragraphs[7], "Tên viết tắt: DASP-v5 / DASP-NIDS")
    replace_text_keep_first_run(cover_paragraphs[8], "Phiên bản cập nhật theo mã nguồn và kết quả thực nghiệm đến ngày 22/07/2026")
    cover_paragraphs[8].alignment = WD_ALIGN_PARAGRAPH.CENTER

    remove_body_from(document, "1. Đặt vấn đề")
    document.add_page_break()

    add_heading(document, "1. Đặt vấn đề và phạm vi nghiên cứu", 1)
    add_body(document, "Mô hình phát hiện xâm nhập mạng triển khai theo luồng phải dự đoán trước khi nhãn của cửa sổ hiện tại được phép tham gia cập nhật. Khi phân phối lưu lượng thay đổi, cập nhật Online-MLP thông thường có thể học được mẫu mới nhưng cũng có thể làm suy giảm kiến thức đã tích lũy. Ngược lại, một chính sách bảo vệ quá mạnh có thể làm chậm khả năng phục hồi sau drift.")
    add_body(document, "Nghiên cứu hiện tại không còn xem tín hiệu dịch chuyển biểu diễn là bằng chứng đủ để cập nhật mạnh một tầng. DASP-v5 tách ba quyết định: MMD chỉ phát cảnh báo không giám sát; counterfactual probe dùng nhãn trễ để ước lượng lợi ích và chi phí quên; dual guard quyết định candidate nào đủ an toàn để triển khai. Nếu không có cảnh báo được xác nhận hoặc không candidate nào vượt guard, hệ thống quay về cập nhật Online-MLP thông thường.")
    add_equation(document, "Predict → MMD alarm → delayed-label utility probes → candidate portfolio → dual guard → commit or Online fallback")

    add_heading(document, "2. Những nội dung được cập nhật so với đề cương cũ", 1)
    add_table(
        document,
        ["Thành phần", "Đề cương cũ", "DASP-v5 đang triển khai"],
        [
            ["Vai trò của MMD", "Vừa báo drift vừa phân bổ plasticity", "Chỉ phát cảnh báo; không trực tiếp quyết định plasticity"],
            ["Tín hiệu phân bổ", "Representation shift + gradient conflict", "Recent gain − λ × memory cost từ one-step counterfactual probe"],
            ["Candidate", "Online và một DASP candidate", "Online, global-utility và layer-utility từ cùng trạng thái trước cập nhật"],
            ["Safety gate", "Một recent LCB, z = 1.645", "Recent LCB và memory LCB độc lập, z = 1.96; có tolerance cho recent guard"],
            ["ηfloor", "0.10", "0.25"],
            ["Claim trung tâm", "Layer-wise > Global", "Guarded portfolio cải thiện Online trong một số drift; layer-wise > global chưa được chứng minh"],
            ["Dữ liệu", "Dự kiến thêm LUFlow và cross-dataset", "S3, S0, NF-CICIDS và NF-UNSW theo đúng các stream đã chạy"],
        ],
        widths=[1.25, 2.25, 3.0],
    )

    add_heading(document, "3. Khoảng trống nghiên cứu", 1)
    add_body(document, "Phần lớn cơ chế adaptation hoặc cập nhật vô điều kiện khi phát hiện drift, hoặc áp dụng một cơ chế bảo vệ cố định cho toàn mô hình. Hai điểm còn thiếu đối với Online-MLP trong NIDS là: (i) tách cảnh báo drift khỏi quyết định triển khai cập nhật; và (ii) định lượng trực tiếp trade-off giữa cải thiện trên dữ liệu gần đây và suy giảm trên dữ liệu lịch sử trước khi commit.")
    add_body(document, "Khoảng trống mà DASP-v5 nhắm tới vì vậy không phải là chứng minh mọi tầng luôn cần mức học khác nhau. Mục tiêu hẹp và kiểm chứng được là xây dựng một adaptation portfolio có fallback bảo thủ, trong đó candidate thích nghi chỉ được chọn khi bằng chứng có nhãn trễ cho thấy utility dương dưới bất định lấy mẫu.")

    add_heading(document, "4. Câu hỏi nghiên cứu", 1)
    add_table(
        document,
        ["Mã", "Câu hỏi nghiên cứu đã cập nhật", "Bằng chứng cần dùng"],
        [
            ["RQ1", "DASP-v5 có cải thiện post-warm-up macro-F1 so với Online-MLP dưới controlled recurring drift hay không?", "S3, 20 paired seeds, CI/effect size/Wilcoxon-Holm"],
            ["RQ2", "Cơ chế báo động và guard có tránh tạo penalty trên stationary stream S0 hay không?", "S0, 10 paired seeds; macro-F1, recall và update time"],
            ["RQ3", "DASP-v5 xếp hạng ở đâu so với các neural continual-learning và drift-adaptive NIDS baselines?", "Benchmark 8 phương pháp trên S3, S0 và hai NetFlow streams"],
            ["RQ4", "Mức cải thiện đạt được với trade-off tính toán và thời gian cập nhật nào?", "Cumulative update time, gradient evaluations và hiệu quả theo stream"],
        ],
        widths=[0.5, 4.2, 1.8],
    )

    add_heading(document, "5. Giả thuyết và tiêu chí diễn giải", 1)
    add_heading(document, "H1 — Cải thiện có điều kiện trên recurring drift", 2)
    add_body(document, "DASP-v5 được kỳ vọng có macro-F1 sau warm-up cao hơn Online-MLP trên S3 vì candidate portfolio chỉ rời khỏi đường cập nhật Online khi utility có bằng chứng dương.")
    add_heading(document, "H2 — Không có penalty quan sát được trên negative control", 2)
    add_body(document, "Trên S0, DASP-v5 được kỳ vọng tái tạo đường cập nhật Online khi không có alarm đủ điều kiện. Kết luận chỉ áp dụng cho stationary control đã thử, không được diễn giải thành bảo đảm no-harm phổ quát.")
    add_heading(document, "H3 — Layer-specific utility là giả thuyết cơ chế, không phải kết luận đã xác nhận", 2)
    add_body(document, "So sánh locked seeds 101–110 chưa cho thấy layer-utility hoặc portfolio vượt global-utility. Do đó, đề cương mới giữ layer candidate như một thành phần được lựa chọn bởi guard, nhưng loại bỏ mệnh đề bắt buộc “Layer-wise > Global” khỏi claim trung tâm.")
    add_heading(document, "H4 — Không giả định ưu thế phổ quát", 2)
    add_body(document, "DASP-v5 phải được đánh giá cùng Replay, EWC, A-GEM và các NIDS-specific methods gồm SSF, RepShield và CDDA-MD. Nếu một baseline mạnh hơn, kết quả đó được giữ nguyên để xác định miền áp dụng, thay vì loại khỏi benchmark.")

    add_heading(document, "6. Mục tiêu và đóng góp khoa học", 1)
    add_bullet(document, "Thiết kế một cơ chế candidate-based adaptation tách cảnh báo không giám sát khỏi quyết định commit có giám sát.")
    add_bullet(document, "Đề xuất counterfactual layer utility đo recent gain đồng thời với historical forgetting cost.")
    add_bullet(document, "Đề xuất dual lower-confidence-bound guard kiểm soát riêng lợi ích gần đây và chi phí trên memory.")
    add_bullet(document, "Thiết kế benchmark cuối gồm 320 runs cho 8 phương pháp chính và 60 runs bổ sung cho ablation, kèm negative control, chronological external streams, paired inference và công bố cả kết quả âm.")
    add_body(document, "Claim được hỗ trợ hiện tại: guarded utility-portfolio adaptation cải thiện một Online-MLP và một số neural continual-learning baselines trên controlled A–B–A recurrence. Claim chưa được hỗ trợ: DASP-v5 tốt hơn Replay hoặc các NIDS-specific baselines mới trên mọi stream; layer-wise utility tốt hơn global utility một cách nội tại.")

    add_heading(document, "7. Mô hình nền và giao thức online", 1)
    add_body(document, "Mô hình nền là MLP ba hidden blocks: 256–128–64, mỗi block gồm Linear, LayerNorm và ReLU, sau đó là classifier. Huấn luyện sử dụng Adam-style gradient update với learning rate 0.001, weight decay 0.0001, batch size 256 và một epoch cho mỗi cửa sổ.")
    add_equation(document, "Input(d) → [Linear–LayerNorm–ReLU]256 → [Linear–LayerNorm–ReLU]128 → [Linear–LayerNorm–ReLU]64 → Classifier")
    add_body(document, "Giao thức prequential là test-then-train với label delay bằng một cửa sổ. Mỗi cửa sổ được dự đoán và ghi metric trước khi nhãn được tiết lộ cho cập nhật. Dữ liệu trong cùng stream không được shuffle theo thời gian và scaler không được fit bằng dữ liệu tương lai.")
    add_equation(document, "predict(Wt; θt) → observe yt−1 → update(θt, Wt−1, yt−1)")

    add_heading(document, "8. Tổng quan kiến trúc DASP-v5", 1)
    add_figure(document, "fig1_architecture.png", "Hình 1. Luồng xử lý DASP-v5: MMD chỉ phát alarm; delayed labels tạo utility probes và candidate portfolio; dual guard quyết định commit.", width=6.15)
    add_body(document, "Trong cửa sổ bình thường, DASP-v5 trùng với Online-MLP. Khi layer-representation MMD vượt ngưỡng trong P = 2 cửa sổ liên tiếp, hệ thống tạo ba candidate từ cùng pre-update state. Quyết định cuối không dùng MMD mà dùng loss difference trên recent guard và memory guard.")

    add_heading(document, "9. Phát cảnh báo bằng representation MMD", 1)
    add_body(document, "Một frozen reference model ánh xạ reference window và current window sang hidden representations. Với mỗi layer ℓ, hệ thống tính MMD và chuẩn hóa bằng median/MAD thu được trong calibration windows. Severity là trung bình top-k normalized shifts. Alarm được xác nhận khi severity vượt ngưỡng trong hai cửa sổ liên tiếp.")
    add_equation(document, "dₜ⁽ˡ⁾ = MMD(Href⁽ˡ⁾, Ht⁽ˡ⁾);   severityₜ = mean(top-k{normalized dₜ⁽ˡ⁾})")
    add_body(document, "MMD phản ánh marginal representation shift, không chứng minh P(Y|X) đã thay đổi và không bảo đảm một adaptive update sẽ cải thiện dự đoán. Vì vậy, giá trị MMD không xuất hiện trong công thức utility hoặc điều kiện commit.")

    add_heading(document, "10. Counterfactual layer utility", 1)
    add_body(document, "Tại một confirmed alarm, mỗi layer được thử bằng một one-step hypothetical update trên phần train của cửa sổ có nhãn trễ. Candidate thăm dò này không được triển khai. Loss trước và sau bước thử được đo trên recent guard và một memory probe độc lập.")
    add_equation(document, "gainₗ = Lrecent(θ) − Lrecent(θ′ₗ)")
    add_equation(document, "costₗ = max{0, Lmemory(θ′ₗ) − Lmemory(θ)}")
    add_equation(document, "utilityₗ = gainₗ − λf · costₗ")
    add_body(document, "Chỉ phần utility dương được chuẩn hóa thành plasticity. Với ηfloor = 0.25, layer có utility thấp vẫn nhận một phần learning rate cơ sở; layer có utility cao được phép tiến gần learning rate đầy đủ. Memory có tối đa 5,000 mẫu; utility probe và memory guard dùng tối đa 256 mẫu.")
    add_equation(document, "pₗ = normalize(max{0, utilityₗ});   ηₗ = ηbase[0.25 + 0.75pₗ]")

    add_heading(document, "11. Candidate portfolio và gradient protection", 1)
    add_table(
        document,
        ["Candidate", "Cách tạo", "Vai trò"],
        [
            ["Online", "Cập nhật Online-MLP chuẩn", "Fallback bắt buộc và comparator tại alarm"],
            ["Global-utility", "Dùng mean(pₗ) cho toàn bộ layer", "Kiểm tra utility chung có đủ tốt hay không"],
            ["Layer-utility", "Mỗi layer dùng pₗ riêng", "Cho phép phân bổ mức thích nghi không đồng nhất"],
        ],
        widths=[1.3, 3.0, 2.2],
    )
    add_body(document, "Ba candidate được tạo từ cùng trạng thái θ trước cập nhật để tránh lợi thế do thứ tự. Trong các adaptive candidates, nếu recent gradient và memory gradient xung đột, thành phần gradient chống lại memory được giảm theo plasticity. Gradient protection là cơ chế hỗ trợ cập nhật, không phải tín hiệu dùng để ước lượng plasticity.")

    add_heading(document, "12. Dual guard và quyết định commit", 1)
    add_body(document, "Với từng candidate thích nghi, loss difference được tính trên từng mẫu của recent guard và memory guard so với Online candidate. Giá trị dương có nghĩa candidate thích nghi tốt hơn Online. Mỗi guard dùng lower confidence bound (LCB) với z = 1.96.")
    add_equation(document, "Δrecent,i = CEonline,i − CEcandidate,i;   LCBrecent = mean(Δrecent) − 1.96·SE(Δrecent)")
    add_equation(document, "Δmemory,i = CEonline,i − CEcandidate,i;   LCBmemory = mean(Δmemory) − 1.96·SE(Δmemory)")
    add_equation(document, "eligible ⇔ LCBrecent ≥ −0.02  and  LCBrecent + wm·LCBmemory > 0")
    add_body(document, "Nếu nhiều candidate đủ điều kiện, hệ thống chọn candidate có combined LCB lớn nhất. Nếu không candidate nào đủ điều kiện, Online được commit. Cơ chế này giải quyết drift bằng cách thử các hướng cập nhật có kiểm soát, không phải bằng cách mặc định tái huấn luyện toàn mạng ngay sau alarm.")

    add_heading(document, "13. Mã giả thuật toán", 1)
    add_table(
        document,
        ["Dòng", "DASP-v5 cho mỗi cửa sổ Wt"],
        [
            ["1", "Predict Wt bằng θt và ghi metric trước cập nhật."],
            ["2", "Tính layer MMD bằng frozen reference; cập nhật alarm persistence."],
            ["3", "Khi nhãn của Wt−1 đến, chia 80% train và 20% recent guard."],
            ["4", "Nếu alarm chưa xác nhận: cập nhật Online; cập nhật memory; tiếp tục."],
            ["5", "Nếu alarm xác nhận: với mỗi layer ℓ, thực hiện one-step counterfactual probe."],
            ["6", "Tính gainℓ, costℓ, utilityℓ và plasticity pℓ."],
            ["7", "Từ cùng θt tạo Online, Global-utility và Layer-utility candidates."],
            ["8", "Tính recent LCB và memory LCB cho từng adaptive candidate so với Online."],
            ["9", "Lọc candidate theo recent tolerance và combined LCB > 0."],
            ["10", "Commit candidate eligible có combined LCB lớn nhất; nếu rỗng, commit Online."],
            ["11", "Cập nhật bounded memory; chuyển sang cửa sổ tiếp theo."],
        ],
        widths=[0.6, 5.9],
    )

    document.add_page_break()
    add_heading(document, "14. Thiết kế thực nghiệm đã chốt", 1)
    add_heading(document, "14.1. Kịch bản và ma trận chạy", 2)
    add_table(
        document,
        ["Stream", "Thiết kế", "Seeds/method", "Số phương pháp", "Runs"],
        [
            ["Synthetic S3", "A1(Hulk) windows 0–7 → B(DDoS) 8–15 → A2(Hulk) 16–23; 24×1,024; warm-up 4", "20 (201–220)", "8", "160"],
            ["Synthetic S0", "Stationary negative control; 16×1,024; warm-up 4", "10 (201–210)", "8", "80"],
            ["NF-CICIDS2018-v3", "36×4,096, chronological, binary, start row 360,000; warm-up 4", "5 (201–205)", "8", "40"],
            ["NF-UNSW-NB15-v3", "36×4,096, chronological, binary, start row 0; warm-up 4", "5 (201–205)", "8", "40"],
            ["Tổng bảng chính", "Bốn stream/scenario", "—", "8", "320"],
        ],
        widths=[1.15, 3.4, 1.0, 0.7, 0.55],
    )
    add_body(document, "Ngoài 320 runs của bảng chính, Global DASP-v5 và Layer-wise DASP-v5 được chạy thêm trên S3 (20 seeds) và S0 (10 seeds), tương ứng 60 runs. Online-MLP và Full DASP-v5 được tái sử dụng từ bảng chính. Tổng kế hoạch cuối là 380 runs. Locked confirmation cũ với seeds 101–110 chỉ được xem là bằng chứng phát triển, không thay thế ma trận cuối.")

    add_heading(document, "14.2. Thuật toán so sánh", 2)
    add_table(
        document,
        ["Nhóm", "Phương pháp", "Mục đích / trạng thái"],
        [
            ["Online neural", "Online-MLP", "Baseline trung tâm; đã có trong runner"],
            ["Rehearsal", "Replay", "Baseline mạnh cho recurrence; đã có trong runner"],
            ["Regularization", "Online EWC", "Bảo vệ tham số quan trọng; đã có trong runner"],
            ["Gradient constraint", "A-GEM", "Ràng buộc gradient bằng episodic memory; đã có trong runner"],
            ["Đề xuất", "Full DASP-v5", "Guarded utility portfolio trên Online-MLP; đã có trong runner"],
            ["NIDS-specific CL", "SSF", "Strategic selection and forgetting; cần tích hợp và kiểm tra tái lập"],
            ["NIDS-specific CL", "RepShield", "Drift detection + buffer optimization + gradient projection; cần reimplementation có kiểm chứng"],
            ["Drift-adaptive NIDS", "CDDA-MD", "Concept-drift detection and incremental adaptation; cần independent reimplementation"],
        ],
        widths=[1.4, 1.8, 3.2],
    )

    add_heading(document, "14.3. Ablation của DASP-v5", 2)
    add_table(
        document,
        ["Biến thể", "Cơ chế", "Câu hỏi được kiểm tra"],
        [
            ["Online-MLP", "Không có utility allocation hoặc guard", "Adaptation có tốt hơn cập nhật online thông thường?"],
            ["Global DASP-v5", "Một plasticity chung từ mean layer utility", "Utility chung có đủ để giải thích cải thiện?"],
            ["Layer-wise DASP-v5", "Plasticity riêng cho từng layer", "Phân bổ theo layer có hơn global hay không?"],
            ["Full DASP-v5", "Online/global/layer portfolio + dual guard", "Candidate portfolio và guard có tạo thêm lợi ích?"],
        ],
        widths=[1.6, 2.5, 2.3],
    )
    add_body(document, "Không đặt trước kết luận Layer-wise > Global. Ba so sánh Full–Global, Full–Layer và Layer–Global được kiểm định hai phía và hiệu chỉnh Holm trong một family ablation riêng.")

    add_heading(document, "14.4. Chỉ số và thống kê", 2)
    add_bullet(document, "Chỉ số chính: mean post-warm-up macro-F1.")
    add_bullet(document, "Chỉ số phụ: balanced accuracy, attack recall/FNR, p10/worst-window F1, cumulative metrics, A1→A2 forgetting, update time và gradient evaluations.")
    add_bullet(document, "S3 bảng chính: Friedman test trên 8 phương pháp và 20 paired seeds; kiểm định primary one-sided Full DASP-v5 > Online-MLP; các so sánh Full DASP-v5 với 7 baselines được báo cáo cùng Holm correction.")
    add_bullet(document, "Báo cáo paired bootstrap 95% CI với 5,000 resamples và paired Cohen dz; không chỉ dựa vào p-value.")
    add_bullet(document, "Hai NetFlow streams chỉ có 5 paired seeds trên một segment cố định, vì vậy kết quả được diễn giải thận trọng và chủ yếu mang tính mô tả.")

    add_heading(document, "15. Kết quả sơ bộ và trạng thái benchmark cuối", 1)
    add_body(document, "Các số dưới đây đến từ benchmark phát triển trước khi chốt SSF, RepShield và CDDA-MD. Chúng chỉ chứng minh pipeline và định hướng hiệu quả ban đầu; không được trình bày như kết quả của ma trận 380 runs cuối. Bảng chính cuối chỉ được khóa sau khi ba phương pháp mới được triển khai, kiểm tra tái lập và chạy cùng protocol.")
    add_table(
        document,
        ["Kịch bản", "DASP-v5", "So sánh chính", "Kết luận có thể bảo vệ"],
        [
            ["S3 recurring", "0.9127 ± 0.0180", "+0.01659 vs Online; 95% CI [0.01237, 0.02076]; Holm p = 0.000393; dz = 1.683", "Cải thiện Online, EWC và A-GEM; không vượt Replay"],
            ["S0 stationary", "0.993981 ± 0.002790", "Bằng Online; update time 0.379 vs 0.377 s", "Không quan sát penalty trên control đã thử"],
            ["NF-CICIDS", "0.8568", "+0.00986 vs Online; n = 5; Holm p = 0.3125", "Tín hiệu tích cực nhưng chưa đủ confirmatory"],
            ["NF-UNSW", "0.9936", "Bằng Online", "Không quan sát adaptive commit có lợi"],
        ],
        widths=[1.1, 1.2, 3.1, 2.0],
    )
    add_heading(document, "16. Diễn giải kết quả và giới hạn claim", 1)
    add_body(document, "Sơ bộ, DASP-v5 cải thiện Online-MLP trên S3 với CI dương, Holm-corrected significance và effect size lớn; RQ2 chỉ được hỗ trợ trên S0 đã thử. Replay đạt 0.9391 ± 0.0084 và có forgetting thấp hơn, nên chưa có claim ưu thế trước Replay. RQ3 chỉ được trả lời sau khi SSF, RepShield và CDDA-MD hoàn tất cùng protocol.")
    add_body(document, "Locked confirmation cũng bác bỏ cách diễn giải cũ rằng layer-wise utility phải tốt hơn global utility: global = 0.920570, layer = 0.920595 và portfolio = 0.920295. Portfolio − global = −0.000275, 95% CI [−0.001253, 0.000456]; portfolio − layer = −0.000301, 95% CI [−0.001865, 0.000826]. Kết quả âm này được giữ để tránh chọn giả thuyết theo kết quả.")
    add_body(document, "Về chi phí trên S3, DASP-v5 dùng 1.155 s cumulative update time/run, so với Online 0.580 và Replay 0.738 s. Bảng cuối phải bổ sung cùng phép đo cho EWC, A-GEM, SSF, RepShield và CDDA-MD. Đây là số đo phụ thuộc phần cứng và implementation, không phải kết luận độ phức tạp độc lập nền tảng.")

    add_heading(document, "17. Hạn chế và yêu cầu trước khi nộp bài", 1)
    add_bullet(document, "External validation mới dùng một chronological segment cho mỗi dataset và chỉ năm initialization seeds; chưa đại diện cho nhiều giai đoạn thời gian độc lập.")
    add_bullet(document, "S3 là controlled synthetic recurrence; kết quả không tự động chuyển thành ưu thế trên drift tự nhiên hoặc multiclass NIDS.")
    add_bullet(document, "MMD chỉ phát hiện representation shift; nghiên cứu không tuyên bố nó xác định ground-truth concept drift.")
    add_bullet(document, "Các ngưỡng alarm và guard hiện là fixed configuration; cần sensitivity analysis hoặc preregistration rõ ràng hơn.")
    add_bullet(document, "Mã hiện tại đã tách update RNG và replay RNG để tăng công bằng; toàn bộ ma trận 380 runs phải được chạy bằng cùng phiên bản mã cuối trước khi khóa số liệu nộp bài.")
    add_bullet(document, "SSF, RepShield và CDDA-MD phải vượt replication gate trên thiết lập gốc hoặc sanity benchmark trước khi được đưa vào bảng so sánh chính.")
    add_bullet(document, "Future work: nhiều chronological segments, calibration/precision–recall metrics và NIDS-specific replay.")

    add_heading(document, "18. Kết luận và hướng triển khai tiếp theo", 1)
    add_body(document, "Đề cương cập nhật định vị DASP-v5 là một cơ chế guarded selective adaptation cho Online-MLP, không phải một thuật toán khẳng định mọi layer phải thích nghi khác nhau. Khi drift alarm xuất hiện, hệ thống ước lượng utility bằng nhãn trễ, tạo portfolio candidate, kiểm tra lợi ích gần đây và chi phí quên bằng dual LCB, rồi fallback về Online nếu bằng chứng không đủ mạnh.")
    add_equation(document, "Alarm ≠ Commit;   Commit adaptive candidate only if recent utility is tolerable and combined recent–memory evidence is positive")
    add_body(document, "Claim cuối chỉ được khóa sau benchmark 380 runs với SSF, RepShield và CDDA-MD, công bố kết quả âm và tái lập bằng mã cuối.")

    document.core_properties.title = "Đề cương nghiên cứu DASP-v5 cập nhật"
    document.core_properties.subject = "Guarded utility-portfolio adaptation for online NIDS under concept drift"
    document.core_properties.comments = "Updated to match DASP-v5 implementation and experiments on 2026-07-22."
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
