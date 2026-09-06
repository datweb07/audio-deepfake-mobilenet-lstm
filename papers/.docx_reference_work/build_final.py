import copy
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W}

SOURCE = Path(r"D:\audio-deepfake-mobilenet-lstm\papers\test.docx")
SHORT = Path(r"D:\audio-deepfake-mobilenet-lstm\papers\test_shortened_11_pages.docx")
OUTPUT = Path(r"D:\audio-deepfake-mobilenet-lstm\papers\test_final.docx")


def qn(local):
    return f"{{{W}}}{local}"


def text_of(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def first_run_properties(paragraph):
    props = paragraph.xpath(".//w:r[1]/w:rPr", namespaces=NS)
    return copy.deepcopy(props[0]) if props else None


def set_paragraph_text(paragraph, text):
    ppr = paragraph.find(qn("pPr"))
    run_props = first_run_properties(paragraph)
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    run = etree.SubElement(paragraph, qn("r"))
    if run_props is not None:
        run.append(run_props)
    text_node = etree.SubElement(run, qn("t"))
    if text.startswith(" ") or text.endswith(" "):
        text_node.set(f"{{{XML}}}space", "preserve")
    text_node.text = text


def paragraph_by_prefix(root, prefix):
    matches = [p for p in root.xpath("./w:body/w:p", namespaces=NS) if text_of(p).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph starting {prefix!r}, found {len(matches)}")
    return matches[0]


def replace_body_paragraph(root, prefix, new_text):
    paragraph = paragraph_by_prefix(root, prefix)
    set_paragraph_text(paragraph, new_text)
    return paragraph


def split_related_work_paragraph(root):
    paragraph = paragraph_by_prefix(root, "Reproducible anti-spoofing methodologies")
    first = (
        "Reproducible anti-spoofing methodologies were set forth by ASVspoof and t-DCF [1], [3]."
    )
    second = (
        "While WaveFake and survey works illustrate additional synthesis methods and detection frameworks [4], [5]."
    )
    third = (
        "RawNet2 models waveforms directly, whereas AASIST integrates spectral and temporal graph attention "
        "[7], [8]."
    )
    fourth = (
        "LCNN and SincNet illustrate complementary time-frequency and learned-filter approaches [10], [11]."
    )
    set_paragraph_text(paragraph, first)
    sibling = copy.deepcopy(paragraph)
    set_paragraph_text(sibling, second)
    paragraph.addnext(sibling)
    third_sibling = copy.deepcopy(paragraph)
    set_paragraph_text(third_sibling, third)
    sibling.addnext(third_sibling)
    fourth_sibling = copy.deepcopy(paragraph)
    set_paragraph_text(fourth_sibling, fourth)
    third_sibling.addnext(fourth_sibling)


def split_limitations_paragraph(root):
    paragraph = paragraph_by_prefix(root, "The study mixes ImageNet transfer")
    first = (
        "The study mixes ImageNet transfer (MobileNet/EfficientNet), scratch optimization "
        "(ShuffleNet/MnasNet), and external RawNet2/AASIST checkpoints, so evaluation consistency is not "
        "identical training. Missing speaker/source/generator identifiers permit only checksum-group-disjoint "
        "claims. Robustness uses 100 samples; replay is simulated; no unseen corpus, physical replay, "
        "codec-version seal, or probability calibration exists."
    )
    second = (
        "Runtime was measured on one desktop CPU rather than an edge target, and single checkpoints plus "
        "test-set bootstrap do not replace multi-seed training [25]."
    )
    third = (
        "Future work should first execute full-test stresses and a metadata-rich external corpus [16], [17], "
        "then measure physical replay and target hardware before quantization or streaming studies."
    )
    set_paragraph_text(paragraph, first)
    sibling = copy.deepcopy(paragraph)
    set_paragraph_text(sibling, second)
    paragraph.addnext(sibling)
    third_sibling = copy.deepcopy(paragraph)
    set_paragraph_text(third_sibling, third)
    sibling.addnext(third_sibling)


def table_rows(root):
    for table in root.xpath("./w:body/w:tbl", namespaces=NS):
        for row in table.xpath("./w:tr", namespaces=NS):
            yield row


def set_cell_text(cell, text):
    paragraphs = cell.xpath("./w:p", namespaces=NS)
    if not paragraphs:
        paragraph = etree.SubElement(cell, qn("p"))
    else:
        paragraph = paragraphs[0]
        for extra in paragraphs[1:]:
            cell.remove(extra)
    set_paragraph_text(paragraph, text)


def update_tables(root):
    table1 = root.xpath("./w:body/w:tbl", namespaces=NS)[0]
    table1_map = {
        "ASVspoof": "ASVspoof [2], [3]",
        "RawNet2/AASIST": "RawNet2/AASIST [7], [8]",
        "SSL anti-spoofing": "SSL anti-spoofing [12], [13]",
        "Cross-domain studies": "Cross-domain studies [14], [15]",
    }
    for row in table1.xpath("./w:tr[position()>1]", namespaces=NS):
        cells = row.xpath("./w:tc", namespaces=NS)
        current = text_of(cells[0])
        for prefix, replacement in table1_map.items():
            if current.startswith(prefix):
                set_cell_text(cells[0], replacement)
                break

    table3 = root.xpath("./w:body/w:tbl", namespaces=NS)[2]
    model_citations = {
        "MobileNetV3-LSTM": "MobileNetV3-LSTM [18], [19]",
        "ShuffleNetV2-LSTM": "ShuffleNetV2-LSTM [20], [21]",
        "MnasNet-A1-LSTM": "MnasNet-A1-LSTM [23]",
        "EfficientNet-B0-LSTM": "EfficientNet-B0-LSTM [22]",
        "RawNet2": "RawNet2 [7]",
        "AASIST": "AASIST [8], [9]",
    }
    for row in table3.xpath("./w:tr[position()>1]", namespaces=NS):
        cells = row.xpath("./w:tc", namespaces=NS)
        current = text_of(cells[0]).strip()
        if current in model_citations:
            set_cell_text(cells[0], model_citations[current])


def strip_bookmarks(node):
    for bookmark in node.xpath(".//w:bookmarkStart | .//w:bookmarkEnd", namespaces=NS):
        bookmark.getparent().remove(bookmark)


def flatten_hyperlinks(node):
    for hyperlink in node.xpath(".//w:hyperlink", namespaces=NS):
        parent = hyperlink.getparent()
        index = parent.index(hyperlink)
        children = list(hyperlink)
        parent.remove(hyperlink)
        for child in children:
            parent.insert(index, child)
            index += 1


def remove_numbering(paragraph):
    num_prs = paragraph.xpath("./w:pPr/w:numPr", namespaces=NS)
    for num_pr in num_prs:
        num_pr.getparent().remove(num_pr)


def prepend_reference_number(paragraph, number):
    ppr = paragraph.find(qn("pPr"))
    index = 1 if ppr is not None else 0
    run = etree.Element(qn("r"))
    rpr = etree.SubElement(run, qn("rPr"))
    size = etree.SubElement(rpr, qn("sz"))
    size.set(qn("val"), "16")
    text_node = etree.SubElement(run, qn("t"))
    text_node.set(f"{{{XML}}}space", "preserve")
    text_node.text = f"[{number}] "
    paragraph.insert(index, run)


def compact_reference_paragraph(paragraph):
    ppr = paragraph.find(qn("pPr"))
    if ppr is None:
        ppr = etree.Element(qn("pPr"))
        paragraph.insert(0, ppr)
    spacing = ppr.find(qn("spacing"))
    if spacing is None:
        spacing = etree.SubElement(ppr, qn("spacing"))
    spacing.set(qn("before"), "0")
    spacing.set(qn("after"), "0")
    spacing.set(qn("line"), "160")
    spacing.set(qn("lineRule"), "exact")
    if ppr.find(qn("keepLines")) is None:
        etree.SubElement(ppr, qn("keepLines"))
    for run in paragraph.xpath(".//w:r", namespaces=NS):
        rpr = run.find(qn("rPr"))
        if rpr is None:
            rpr = etree.Element(qn("rPr"))
            run.insert(0, rpr)
        for size_name in ("sz", "szCs"):
            size = rpr.find(qn(size_name))
            if size is None:
                size = etree.SubElement(rpr, qn(size_name))
            size.set(qn("val"), "14")


def merge_reference_paragraphs(first, continuation):
    for child in list(continuation):
        if child.tag != qn("pPr") and child.tag not in {qn("bookmarkStart"), qn("bookmarkEnd")}:
            first.append(copy.deepcopy(child))


def append_references(final_root, source_root):
    source_paragraphs = source_root.xpath("./w:body/w:p", namespaces=NS)
    body = final_root.find(qn("body"))
    sect_pr = body.find(qn("sectPr"))

    heading = copy.deepcopy(source_paragraphs[130])
    strip_bookmarks(heading)
    set_paragraph_text(heading, "References")
    body.insert(body.index(sect_pr), heading)

    source_map = [
        (133, None),       # ASVspoof 2019
        (134, None),       # ASVspoof 2021
        (135, None),       # t-DCF
        (136, 137),        # WaveFake + URL
        (138, None),       # audio-deepfake survey; corrected below
        (139, None),       # speech-deepfake survey
        (140, None),       # RawNet2
        (141, None),       # AASIST
        (142, None),       # graph attention networks
        (145, None),       # LCNN
        (146, None),       # SincNet
        (150, 151),        # wav2vec anti-spoofing
        (152, None),       # calibrated/generalizable SSL
        (153, None),       # generalization
        (154, None),       # attack-agnostic dataset
        (155, None),       # harder or different?
        (156, None),       # cross-domain dataset
        (159, None),       # MobileNetV2
        (160, None),       # MobileNetV3
        (161, None),       # ShuffleNet
        (162, None),       # ShuffleNetV2
        (163, None),       # EfficientNet
        (164, None),       # MnasNet
        (165, None),       # librosa
        (166, None),       # bootstrap
    ]

    for number, (main_index, continuation_index) in enumerate(source_map, 1):
        paragraph = copy.deepcopy(source_paragraphs[main_index])
        if continuation_index is not None:
            merge_reference_paragraphs(paragraph, source_paragraphs[continuation_index])
        strip_bookmarks(paragraph)
        flatten_hyperlinks(paragraph)
        remove_numbering(paragraph)

        if number == 4:
            set_paragraph_text(
                paragraph,
                "J. Frank and L. Schönherr, “WaveFake: A data set to facilitate audio deepfake detection,” "
                "in NeurIPS Datasets and Benchmarks, 2021. [Online]. Available: "
                "https://arxiv.org/abs/2111.02813",
            )
        elif number == 5:
            set_paragraph_text(
                paragraph,
                "J. Yi, C. Wang, J. Tao, X. Zhang, C. Y. Zhang, and Y. Zhao, “Audio deepfake detection: "
                "A survey,” arXiv preprint arXiv:2308.14970, 2023. [Online]. Available: "
                "https://arxiv.org/abs/2308.14970",
            )
        elif number == 12:
            set_paragraph_text(
                paragraph,
                "H. Tak, M. Todisco, X. Wang, J.-w. Jung, J. Yamagishi, and N. Evans, “Automatic speaker "
                "verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation,” "
                "in Odyssey, 2022, pp. 112–119.",
            )

        corrected = text_of(paragraph)
        for old, new in {
            "P. Velickovic": "P. Veličković",
            "P. Lio": "P. Liò",
            "N. M. Muller": "N. M. Müller",
            "K. Bottinger": "K. Böttinger",
            "efficient cnn architecture": "efficient CNN architecture",
            "signal analysis in python": "signal analysis in Python",
        }.items():
            corrected = corrected.replace(old, new)
        if corrected != text_of(paragraph):
            set_paragraph_text(paragraph, corrected)

        prepend_reference_number(paragraph, number)
        compact_reference_paragraph(paragraph)
        body.insert(body.index(sect_pr), paragraph)


def format_references_as_two_columns(root):
    body = root.find(qn("body"))
    final_sect_pr = body.find(qn("sectPr"))
    heading = paragraph_by_prefix(root, "References")
    heading_ppr = heading.find(qn("pPr"))
    if heading_ppr is None:
        heading_ppr = etree.Element(qn("pPr"))
        heading.insert(0, heading_ppr)

    closing_sect_pr = copy.deepcopy(final_sect_pr)
    existing_type = closing_sect_pr.find(qn("type"))
    if existing_type is None:
        existing_type = etree.SubElement(closing_sect_pr, qn("type"))
    existing_type.set(qn("val"), "continuous")
    heading_ppr.append(closing_sect_pr)

    columns = final_sect_pr.find(qn("cols"))
    if columns is None:
        columns = etree.SubElement(final_sect_pr, qn("cols"))
    columns.set(qn("num"), "2")
    columns.set(qn("space"), "360")
    columns.set(qn("equalWidth"), "1")
    section_type = final_sect_pr.find(qn("type"))
    if section_type is None:
        section_type = etree.SubElement(final_sect_pr, qn("type"))
    section_type.set(qn("val"), "continuous")


def build():
    if not SOURCE.exists() or not SHORT.exists():
        raise FileNotFoundError("Required source documents are missing")
    shutil.copy2(SHORT, OUTPUT)
    with zipfile.ZipFile(SOURCE) as source_zip:
        source_root = etree.fromstring(source_zip.read("word/document.xml"))
    with zipfile.ZipFile(OUTPUT) as current_zip:
        files = {name: current_zip.read(name) for name in current_zip.namelist()}
    final_root = etree.fromstring(files["word/document.xml"])

    replace_body_paragraph(
        final_root,
        "Synthetic and converted speech threaten",
        "Synthetic and converted speech threaten speaker verification and human decision making. ASVspoof "
        "established shared logical- and physical-access protocols [1], while WaveFake broadened public "
        "generated-audio resources [4]. Yet reported performance remains difficult to compare when detectors "
        "use opposite class orders, incompatible durations, different preprocessing, or inconsistent runtime "
        "boundaries. Byte-identical files crossing data splits create a separate leakage risk.",
    )
    replace_body_paragraph(
        final_root,
        "LAVA is concerned with the evaluation interface",
        "LAVA is concerned with the evaluation interface rather than making each detector use a single "
        "architecture. It retains the original computation of Mel-sequence, raw waveform, recurrent and graph "
        "attention models while unifying artifact verification, P(FAKE), integrity test, saved output, stress "
        "inputs, timing and aggregations. Current data includes four locally trained lightweight artifacts and "
        "two pretrained artifacts; clean evaluation is full testing, while robustness is diagnostic [5], [6]. "
        "We ask: Three main research questions (RQs) are answered in this paper:",
    )
    split_related_work_paragraph(final_root)
    replace_body_paragraph(
        final_root,
        "MobileNet, ShuffleNet, EfficientNet, and MnasNet pursue efficiency",
        "MobileNet, ShuffleNet, EfficientNet, and MnasNet pursue efficiency through distinct search and "
        "scaling principles [19], [21]. Their published vision or phone measurements are not evidence of LAVA "
        "audio runtime. Table I positions the current contribution: robustness, resource measurement, "
        "integrity, and heterogeneous score semantics are evaluated jointly, while training provenance remains explicit.",
    )
    replace_body_paragraph(
        final_root,
        "Lightweight inputs are decoded",
        "Lightweight inputs are decoded, channel-averaged, polyphase-resampled to 22,050 Hz, and "
        "padded/truncated to 3.0 s. Six chronological 0.5-s segments use a Hann STFT (N = 2048, hop 512), "
        "128 HTK-style Mel filters spanning 20–8,000 Hz, relative dB mapping clipped to 80 dB, [0, 255] "
        "scaling, bilinear 224 × 224 resizing, and RGB replication [24]. For segment t",
    )
    split_limitations_paragraph(final_root)
    update_tables(final_root)
    append_references(final_root, source_root)
    format_references_as_two_columns(final_root)

    files["word/document.xml"] = etree.tostring(
        final_root, xml_declaration=True, encoding="UTF-8", standalone="yes"
    )
    temp_path = OUTPUT.with_suffix(".tmp")
    with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as output_zip:
        for name, data in files.items():
            output_zip.writestr(name, data)
    temp_path.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
