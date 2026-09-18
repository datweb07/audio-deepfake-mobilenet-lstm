from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "PaperID 804 final.docx"
BASE = ROOT / "PaperID 804.docx"
BACKUP = ROOT / "PaperID 804 final_before_last_review.docx"
RESPONSE = ROOT / "Response_to_Reviewers_final.docx"


def effective_size(run, paragraph):
    if run.font.size:
        return round(run.font.size.pt, 2)
    if paragraph.style and paragraph.style.font.size:
        return round(paragraph.style.font.size.pt, 2)
    return None


def doc_stats(path: Path):
    doc = Document(path)
    return {
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "images": len(doc.inline_shapes),
        "equations": len(doc.element.body.findall(".//" + qn("m:oMath")))
        + len(doc.element.body.findall(".//" + qn("m:oMathPara"))),
    }


for path in (PAPER, RESPONSE):
    try:
        with ZipFile(path) as zf:
            bad = zf.testzip()
            print(f"ZIP {path.name}: PASS entries={len(zf.namelist())} bad={bad}")
    except BadZipFile as exc:
        print(f"ZIP {path.name}: FAIL {exc}")

print("accepted", doc_stats(BASE))
print("previous_working", doc_stats(BACKUP))
print("final", doc_stats(PAPER))

doc = Document(PAPER)
title = next(p for p in doc.paragraphs if p.text.strip().startswith("LAVA:"))
abstract = next(p for p in doc.paragraphs if p.text.strip().startswith("Abstract."))
keywords = next(p for p in doc.paragraphs if p.text.strip().startswith("Keywords:"))

for label, paragraph in (("title", title), ("abstract", abstract), ("keywords", keywords)):
    sizes = sorted({effective_size(r, paragraph) for r in paragraph.runs if r.text.strip()})
    print(label, sizes)

body_sizes = Counter()
body_examples = []
for p in doc.paragraphs:
    text = p.text.strip()
    if not text or p in (title, abstract, keywords):
        continue
    if p.style and p.style.name in {"Title", "Author", "Affiliation", "Email", "Heading 1", "Heading 2", "Heading 3", "Heading 4", "Caption", "Reference Item"}:
        continue
    if text in {"Acknowledgement", "References"}:
        continue
    for run in p.runs:
        if not run.text.strip():
            continue
        size = effective_size(run, p)
        body_sizes[size] += len(run.text)
        if size != 10.0 and len(body_examples) < 15:
            body_examples.append((p.style.name if p.style else None, size, text[:100]))
print("body size chars", body_sizes)
print("non-10 body examples", body_examples)

body_text = "\n".join(p.text for p in doc.paragraphs)
for table in doc.tables[:-1]:
    body_text += "\n" + "\n".join(cell.text for row in table.rows for cell in row.cells)

sentence_violations = []
for sentence in re.split(r"(?<=[.!?])\s+", body_text):
    ids = set()
    for group in re.findall(r"\[([0-9,\-\s]+)\]", sentence):
        for part in re.split(r"[,\s]+", group.strip()):
            if not part:
                continue
            if "-" in part:
                a, b = map(int, part.split("-", 1))
                ids.update(range(a, b + 1))
            elif part.isdigit():
                ids.add(int(part))
    if len(ids) > 2:
        sentence_violations.append((sorted(ids), sentence[:180]))
print("citation sentence violations", len(sentence_violations), sentence_violations[:5])

refs = []
for cell in doc.tables[-1].rows[0].cells:
    refs.extend(p.text.strip() for p in cell.paragraphs if re.match(r"^\[\d+\]", p.text.strip()))
years = [int(y) for ref in refs for y in re.findall(r"\b(20\d{2})\b", ref)]
recent = [ref for ref in refs if re.search(r"\b202[4-6]\b", ref)]
print("references", len(refs), "recent_2024_2026", len(recent))
print("recent ids", [re.match(r"^\[(\d+)\]", ref).group(1) for ref in recent])

response = Document(RESPONSE)
response_text = "\n".join(p.text for p in response.paragraphs)
required = [
    "Reviewer Comment 1:",
    "Response:",
    "Revision in Manuscript:",
    "11-page manuscript",
    "References [1]-[24]",
    "Figures 1-6",
    "Partially addressed",
    "Retained as a limitation",
]
print("response markers", {token: token in response_text for token in required})
