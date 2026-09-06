import sys
import zipfile
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}

with zipfile.ZipFile(sys.argv[1]) as archive:
    root = etree.fromstring(archive.read("word/document.xml"))
    paragraphs = root.xpath("./w:body/w:p", namespaces=NS)
    for raw_index in sys.argv[2:]:
        index = int(raw_index)
        print(f"\n--- paragraph {index} ---")
        print(etree.tostring(paragraphs[index], encoding="unicode", pretty_print=True))
