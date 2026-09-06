import re
import sys
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def root(path):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml"))


def visible_main_text(path, stop_at_references):
    document = root(path)
    parts = []
    for child in document.xpath("./w:body/*", namespaces=NS):
        if child.tag == f"{{{W}}}tbl":
            value = " ".join(
                "".join(cell.xpath(".//w:t/text()", namespaces=NS))
                for cell in child.xpath(".//w:tc", namespaces=NS)
            )
        else:
            value = "".join(child.xpath(".//w:t/text()", namespaces=NS))
        if stop_at_references and child.tag == f"{{{W}}}p" and value.strip() == "References":
            break
        parts.append(value)
    combined = " ".join(parts)
    combined = re.sub(r"\[(?:\d+)(?:\s*,\s*\d+)*\]", " ", combined)
    return re.findall(r"[A-Za-zÀ-ỹ0-9]+(?:-[A-Za-zÀ-ỹ0-9]+)*", combined.casefold())


short_tokens = visible_main_text(Path(sys.argv[1]), False)
final_tokens = visible_main_text(Path(sys.argv[2]), True)
if short_tokens != final_tokens:
    for index, (left, right) in enumerate(zip(short_tokens, final_tokens)):
        if left != right:
            raise AssertionError(
                f"content mismatch at token {index}: shortened={left!r}, final={right!r}; "
                f"short context={short_tokens[max(0,index-8):index+9]!r}; "
                f"final context={final_tokens[max(0,index-8):index+9]!r}"
            )
    raise AssertionError(f"token-count mismatch: shortened={len(short_tokens)}, final={len(final_tokens)}")
print(f"unchanged_nonreference_tokens={len(final_tokens)}")
