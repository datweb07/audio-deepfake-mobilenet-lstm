import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def text_of(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def audit(path):
    with zipfile.ZipFile(path) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
        paragraphs = []
        for index, paragraph in enumerate(root.xpath("./w:body/w:p", namespaces=NS)):
            styles = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
            fields = paragraph.xpath(".//w:instrText/text() | .//@w:instr", namespaces=NS)
            paragraphs.append(
                {
                    "index": index,
                    "style": styles[0] if styles else "",
                    "text": text_of(paragraph),
                    "fields": fields,
                    "citations": [int(value) for value in re.findall(r"\[(\d{1,2})\]", text_of(paragraph))],
                }
            )
        return {
            "file": str(path),
            "parts": archive.namelist(),
            "paragraphs": paragraphs,
            "tables": [
                [[text_of(cell) for cell in row.xpath("./w:tc", namespaces=NS)] for row in table.xpath("./w:tr", namespaces=NS)]
                for table in root.xpath("./w:body/w:tbl", namespaces=NS)
            ],
        }


if __name__ == "__main__":
    refs_only = "--refs" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--refs"]
    for arg in args:
        result = audit(Path(arg))
        print("\n===", Path(arg).name, "===")
        print("special parts:", [p for p in result["parts"] if "customXml" in p or "endnote" in p.lower()])
        print("body paragraphs:", len(result["paragraphs"]), "tables:", len(result["tables"]))
        started_refs = not refs_only
        for paragraph in result["paragraphs"]:
            if paragraph["text"].strip().lower() == "references":
                started_refs = True
            if not started_refs:
                continue
            if paragraph["text"] or paragraph["style"] or paragraph["fields"]:
                print(f'{paragraph["index"]:03d} [{paragraph["style"]}] {paragraph["text"]}')
                if paragraph["fields"]:
                    print("    FIELDS", json.dumps(paragraph["fields"], ensure_ascii=False))
        print("\nTABLES")
        for index, table in enumerate(result["tables"], 1):
            print("TABLE", index)
            for row in table:
                print(" | ".join(row))
