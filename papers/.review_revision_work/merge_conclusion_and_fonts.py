from copy import deepcopy
from pathlib import Path
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(r"D:\audio-deepfake-mobilenet-lstm\papers")
PAPER = ROOT / "PaperID 804 final.docx"
RESPONSE = ROOT / "Response_to_Reviewers_final.docx"


def backup_once(path: Path, suffix: str) -> None:
    backup = path.with_name(path.stem + suffix + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)


def set_rfonts(rpr) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{key}"), "Times New Roman")
    for key in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        attr = qn(f"w:{key}")
        if attr in rfonts.attrib:
            del rfonts.attrib[attr]


def force_times_new_roman(doc: Document) -> None:
    # Normalize style defaults so inherited text also resolves to Times New Roman.
    for style in doc.styles:
        if not hasattr(style, "element"):
            continue
        rpr = style.element.get_or_add_rPr()
        set_rfonts(rpr)

    # Normalize every text run in the main document, tables, hyperlinks, headers,
    # footers, text boxes, footnotes/endnotes that are represented by w:r.
    roots = [doc.element]
    for section in doc.sections:
        roots.extend([section.header._element, section.footer._element])
    seen = set()
    for root in roots:
        if id(root) in seen:
            continue
        seen.add(id(root))
        for run in root.xpath(".//w:r"):
            rpr = run.find(qn("w:rPr"))
            if rpr is None:
                rpr = OxmlElement("w:rPr")
                run.insert(0, rpr)
            set_rfonts(rpr)


def find_exact(doc: Document, text: str):
    matches = [p for p in doc.paragraphs if p.text.strip() == text]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph {text!r}, found {len(matches)}")
    return matches[0]


def replace_paragraph_text(paragraph, text: str) -> None:
    # Headings here contain plain text only. Preserve paragraph properties/style.
    ppr = deepcopy(paragraph._p.pPr) if paragraph._p.pPr is not None else None
    for child in list(paragraph._p):
        paragraph._p.remove(child)
    if ppr is not None:
        paragraph._p.append(ppr)
    paragraph.add_run(text)


def merge_paper_sections() -> None:
    doc = Document(PAPER)
    societal = find_exact(doc, "Practical and Societal Implications")
    limitations = find_exact(doc, "Limitations")
    conclusion = find_exact(doc, "Conclusion")

    # Consolidate former Sections 5--7 into one Section 5 while retaining the
    # reviewer-required limitations and societal-impact content as explicit subsections.
    replace_paragraph_text(societal, "Conclusion")
    societal.style = doc.styles["heading1"]
    paragraphs = doc.paragraphs
    societal_index = next(
        i for i, paragraph in enumerate(paragraphs) if paragraph._p is societal._p
    )
    first_societal_body = paragraphs[societal_index + 1]
    first_societal_body.insert_paragraph_before(
        "Practical and Societal Implications", style="heading2"
    )

    limitations.style = doc.styles["heading2"]
    replace_paragraph_text(conclusion, "Concluding Remarks")
    conclusion.style = doc.styles["heading2"]

    force_times_new_roman(doc)
    doc.save(PAPER)


def synchronize_response() -> None:
    doc = Document(RESPONSE)
    replacements = {
        "Revision in Manuscript: Abstract; Section 3.6; Section 4.2; Section 6, Limitations; Section 7, Conclusion.":
            "Revision in Manuscript: Abstract; Section 3.6; Section 4.2; Section 5.2, Limitations; Section 5.3, Concluding Remarks.",
        "Revision in Manuscript: Section 3.7; Section 4.1; Table 3; Section 6, Limitations; Section 7, Conclusion.":
            "Revision in Manuscript: Section 3.7; Section 4.1; Table 3; Section 5.2, Limitations; Section 5.3, Concluding Remarks.",
        "Response: Fully addressed by moving and consolidating the limitations into a dedicated section immediately before the Conclusion.":
            "Response: Fully addressed by retaining a dedicated Limitations subsection immediately before the concluding remarks within the consolidated Conclusion section.",
        "Revision in Manuscript: Section 6, Limitations.":
            "Revision in Manuscript: Section 5.2, Limitations.",
        "Revision in Manuscript: Section 5, Practical and Societal Implications.":
            "Revision in Manuscript: Section 5.1, Practical and Societal Implications.",
    }
    counts = {key: 0 for key in replacements}
    for paragraph in doc.paragraphs:
        source = paragraph.text.strip()
        if source in replacements:
            replace_paragraph_text(paragraph, replacements[source])
            counts[source] += 1
    missing = [key for key, count in counts.items() if count != 1]
    if missing:
        raise RuntimeError(f"Response synchronization mismatch: {missing}")
    force_times_new_roman(doc)
    doc.save(RESPONSE)


if __name__ == "__main__":
    backup_once(PAPER, "_before_conclusion_merge")
    backup_once(RESPONSE, "_before_conclusion_merge")
    merge_paper_sections()
    synchronize_response()
    print("Updated manuscript structure and normalized both documents to Times New Roman.")
