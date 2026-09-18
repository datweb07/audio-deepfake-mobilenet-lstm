from pathlib import Path
import re
from zipfile import ZipFile

from docx import Document
from PIL import Image

ROOT = Path(r"D:\audio-deepfake-mobilenet-lstm\papers")


def all_text(doc):
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def citation_ids(text):
    """Return only bibliography identifiers 1..27, ignoring numeric data ranges."""
    ids = []
    for match in re.finditer(r"\[([0-9,\-–\s]+)\]", text):
        values = []
        for item in re.split(r"[,\s]+", match.group(1).replace("–", "-").strip()):
            if item.isdigit():
                values.append(int(item))
            elif "-" in item:
                lo, hi = item.split("-", 1)
                if lo.isdigit() and hi.isdigit():
                    values.extend(range(int(lo), int(hi) + 1))
        ids.extend(value for value in values if 1 <= value <= 27)
    return ids


for name in ["PaperID 804 final.docx", "Response_to_Reviewers_final.docx"]:
    path = ROOT / name
    doc = Document(path)
    text = all_text(doc)
    print(f"\n=== {name} ===")
    print("paragraphs", len(doc.paragraphs), "tables", len(doc.tables), "inline_shapes", len(doc.inline_shapes), "sections", len(doc.sections))
    print("table_dims", [(len(t.rows), len(t.columns)) for t in doc.tables])
    print("headings")
    for p in doc.paragraphs:
        if p.style and (p.style.name.startswith("Heading") or re.match(r"^\d+(?:\.\d+)*\s", p.text.strip())):
            print(" -", p.style.name, p.text[:140])
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
        print("OMML equations", xml.count(b"<m:oMath"), "drawing nodes", xml.count(b"<w:drawing"))
        media = sorted(n for n in zf.namelist() if n.startswith("word/media/"))
        print("media", len(media))
        for media_name in media:
            try:
                with zf.open(media_name) as handle:
                    im = Image.open(handle)
                    print(" ", media_name, im.size, im.info.get("dpi"))
            except Exception:
                print(" ", media_name, "non-raster")
    if name.startswith("PaperID"):
        ref_idx = next((i for i,p in enumerate(doc.paragraphs) if p.text.strip() == "References"), None)
        body = "\n".join(p.text for p in doc.paragraphs[:ref_idx]) if ref_idx is not None else text
        refs = doc.paragraphs[ref_idx+1:] if ref_idx is not None else []
        cited = set(citation_ids(body))
        print("reference paragraphs", len([p for p in refs if p.text.strip()]))
        print("cited ids", sorted(cited))
        print("missing citation ids 1..27", sorted(set(range(1,28)) - cited))
        citation_violations = []
        for paragraph_index, paragraph in enumerate(doc.paragraphs[:ref_idx], 1):
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph.text.strip()):
                sentence_ids = citation_ids(sentence)
                if len(sentence_ids) > 2:
                    citation_violations.append((paragraph_index, sentence_ids, sentence))
        print("sentences with >2 citations", len(citation_violations))
        for paragraph_index, sentence_ids, sentence in citation_violations:
            print(" citation violation", paragraph_index, sentence_ids, sentence[:220])
        for idx, table in enumerate(doc.tables, 1):
            print(f"TABLE {idx}")
            for row in table.rows[:8]: print(" | ".join(cell.text.replace("\n", " / ") for cell in row.cells))
