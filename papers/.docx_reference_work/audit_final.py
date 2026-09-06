import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def load(path):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml"))


def text(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def main(path):
    root = load(path)
    paragraphs = root.xpath("./w:body/w:p", namespaces=NS)
    tables = root.xpath("./w:body/w:tbl", namespaces=NS)
    ref_head = next(i for i, p in enumerate(paragraphs) if text(p).strip() == "References")
    body_nodes = paragraphs[:ref_head]
    body_nodes += root.xpath("./w:body/w:tbl//w:tc/w:p", namespaces=NS)
    body_citations = []
    max_per_paragraph = 0
    offenders = []
    for node in body_nodes:
        citations = [int(n) for n in re.findall(r"\[(\d+)\]", text(node))]
        body_citations.extend(citations)
        max_per_paragraph = max(max_per_paragraph, len(citations))
        if len(citations) > 2:
            offenders.append(text(node))

    refs = []
    for paragraph in paragraphs[ref_head + 1:]:
        match = re.match(r"\[(\d+)\]\s", text(paragraph))
        if match:
            refs.append(int(match.group(1)))

    expected = list(range(1, 26))
    counts = Counter(body_citations)
    assert refs == expected, (refs, expected)
    assert sorted(counts) == expected, sorted(counts)
    assert max_per_paragraph <= 2, offenders
    assert len(tables) == 6, len(tables)
    drawing_count = len(root.xpath(".//w:drawing", namespaces=NS))
    assert drawing_count == 9, drawing_count
    print(f"references={len(refs)}")
    print(f"cited_reference_ids={len(counts)}")
    print(f"max_citations_per_paragraph={max_per_paragraph}")
    print(f"tables={len(tables)}")
    print(f"drawings={drawing_count}")
    print("citation_counts=" + ",".join(f"{n}:{counts[n]}" for n in expected))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
