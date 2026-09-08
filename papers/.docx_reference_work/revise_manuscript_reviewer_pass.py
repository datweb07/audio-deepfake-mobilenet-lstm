from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(r"D:\audio-deepfake-mobilenet-lstm")
SOURCE = ROOT / "papers" / "ManuScript-ANTT-ST3-Nhóm07.docx"
OUTPUT = ROOT / "papers" / "ManuScript-ANTT-ST3-Nhom07_revised.docx"
BACKUP = ROOT / "papers" / "ManuScript-ANTT-ST3-Nhóm07_before_reviewer_revision.docx"


def set_paragraph_text(paragraph, text: str) -> None:
    p = paragraph._p
    first_rpr = None
    for child in p:
        if child.tag == qn("w:r"):
            rpr = child.find(qn("w:rPr"))
            if rpr is not None:
                first_rpr = deepcopy(rpr)
                break
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    run = OxmlElement("w:r")
    if first_rpr is not None:
        run.append(first_rpr)
    node = OxmlElement("w:t")
    node.set(qn("xml:space"), "preserve")
    node.text = text
    run.append(node)
    p.append(run)


def set_cell_text(cell, text: str) -> None:
    paragraph = cell.paragraphs[0]
    set_paragraph_text(paragraph, text)
    for extra in list(cell.paragraphs[1:]):
        extra._element.getparent().remove(extra._element)


def find_paragraph(doc: Document, prefix: str):
    matches = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one paragraph beginning {prefix!r}, found {len(matches)}")
    return matches[0]


def insert_paragraph_after(element, template_paragraph, text: str):
    new_p = deepcopy(template_paragraph._p)
    element.addnext(new_p)
    proxy = template_paragraph.__class__(new_p, template_paragraph._parent)
    set_paragraph_text(proxy, text)
    return proxy


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not BACKUP.exists():
        shutil.copy2(SOURCE, BACKUP)

    doc = Document(SOURCE)

    caption = find_paragraph(doc, "Table 1.")
    related_heading = find_paragraph(doc, "Related Work")
    benchmark_p = find_paragraph(doc, "ASVspoof 2019 and 2021")
    generalization_p = find_paragraph(doc, "Studies of unseen attacks")
    old_gap_p = find_paragraph(doc, "As summarized in Table 1")
    table1 = doc.tables[0]

    # Section 2 is reorganized into the three reviewer-requested evidence streams.
    set_paragraph_text(
        benchmark_p,
        "Deepfake detection and benchmark protocols. ASVspoof established shared logical-access, "
        "physical-access, and deepfake tasks covering EER/t-DCF, unknown attacks, replay, and later "
        "channel/compression variation [3], [5]. WaveFake supplies multi-generator synthetic speech; "
        "representative detectors include raw-waveform and graph-attention systems [4], [9].",
    )
    set_paragraph_text(
        generalization_p,
        "Robustness, replay, and unseen-data generalization. Strong in-domain discrimination need not "
        "transfer to unseen generators, datasets, or recording domains [13], [16]. Replay evidence is also "
        "protocol-dependent: physical-access evaluation and simulation are not interchangeable, so claims "
        "must identify the distortion, subset, and replay mode [5].",
    )
    set_paragraph_text(
        old_gap_p,
        "Computational efficiency and deployment-oriented evaluation. Efficient CNN families were designed "
        "under resource constraints, but their vision/mobile measurements do not establish audio-pipeline "
        "latency, throughput, memory, or real-time factor [17], [22]. Such evidence requires explicit "
        "preprocessing, hardware, batch-size, warm-up, and timing boundaries.",
    )

    # Move Table 1 from the Introduction into Related Work, immediately before the gap synthesis.
    caption_el = caption._p
    table_el = table1._tbl
    caption_el.getparent().remove(caption_el)
    table_el.getparent().remove(table_el)
    old_gap_p._p.addnext(caption_el)
    caption_el.addnext(table_el)
    gap = insert_paragraph_after(
        table_el,
        old_gap_p,
        "Research gap. The surveyed studies report clean discrimination, robustness/generalization, or "
        "efficiency under different protocols and boundaries, complicating deployment-oriented comparison "
        "of heterogeneous artifacts. LAVA addresses this measurement and traceability problem through a "
        "common evaluation contract, without claiming unseen-corpus or edge-device validation.",
    )

    set_paragraph_text(
        caption,
        "Table 1. Scope explicitly reported by representative studies. Yes = evaluated; Sim. = simulated "
        "replay; – = not reported. Unseen includes held-out attacks, generators, or domains; ASVspoof replay "
        "entries denote physical-access protocols, whereas LAVA uses simulation only.",
    )

    table1_values = [
        ["Study", "Noise", "Compression", "Replay", "Unseen Data", "Latency/RTF", "Edge"],
        ["ASVspoof 2019 [3]", "–", "–", "Yes", "Yes", "–", "–"],
        ["WaveFake [4]", "–", "–", "–", "–", "–", "–"],
        ["ASVspoof 2021 [5]", "–", "Yes", "Yes", "Yes", "–", "–"],
        ["RawNet2 / AASIST [7], [9]", "–", "–", "–", "Yes", "–", "–"],
        ["Generalization studies [12]–[16]", "–", "–", "–", "Yes", "–", "–"],
        ["LAVA (this work)", "Yes", "Yes", "Sim.", "–", "Yes", "–"],
    ]
    while len(table1.rows) > len(table1_values):
        table1._tbl.remove(table1.rows[-1]._tr)
    for row, values in zip(table1.rows, table1_values):
        for cell, value in zip(row.cells, values):
            set_cell_text(cell, value)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(8.5)

    # Correct shifted citation numbers in the detector specification table.
    detector_names = [
        "MobileNetV3-LSTM [17]",
        "ShuffleNetV2-LSTM [18]",
        "MnasNet-A1-LSTM [21]",
        "EfficientNet-B0-LSTM warm-up [22]",
        "RawNet2 [7]",
        "AASIST [9], [24]",
    ]
    for row, name in zip(doc.tables[2].rows[1:], detector_names):
        set_cell_text(row.cells[0], name)

    # Identify the incomplete EfficientNet artifact consistently in all experimental tables.
    for table_index in (3, 4, 5):
        for row in doc.tables[table_index].rows[1:]:
            if row.cells[0].text.strip() == "EfficientNet-B0":
                set_cell_text(row.cells[0], "EffNet-B0 warm-up")

    replacements = {
        "On the fixed 100-recording diagnostic subset, AWGN was":
            "On the fixed 100-recording diagnostic subset, AWGN was the largest observed weakness for the four high-performing lightweight artifacts. Mean noise degradation ranged from 0.3555 for ShuffleNetV2 to 0.8053 for the available EfficientNet-B0 warm-up checkpoint and increased sharply at 5 and 0 dB. Table 5 and Figure 5 report only these diagnostic conditions.",
        "Codec effects were small for the lightweight artifacts":
            "On the fixed diagnostic subset, codec effects were small for the lightweight artifacts (mean ΔF1: 0.0000–0.0233). Under simulated replay, ShuffleNetV2 retained F1 = 0.9500 (ΔF1 = 0.0256), versus ΔF1 values of 0.0620, 0.0641, and 0.1489 for MobileNetV3, the available EfficientNet-B0 warm-up checkpoint, and MnasNet-A1. Negative deltas for the external artifacts reflect score redistribution around weak baselines, not general robustness.",
        "Aggregate failure-pattern analysis.":
            "Aggregate failure-pattern analysis. On the fixed diagnostic subset, low-SNR AWGN produced the largest aggregate degradation; codec effects were limited, while simulated replay affected MnasNet-A1 most and ShuffleNetV2 least. These patterns establish neither causality nor general robustness. Without paired clean/stressed scores, sample-level prediction flips cannot be identified.",
        "Under the common single-thread desktop-CPU protocol":
            "Under the single-thread desktop-CPU protocol in Table 6, MobileNetV3 had the lowest end-to-end latency and RTF; ShuffleNetV2 ranked second among locally trained artifacts. The available EfficientNet-B0 warm-up checkpoint was the slowest Mel-sequence artifact. AASIST paired the smallest file with the largest latency, whereas RawNet2 paired the most parameters with moderate latency; size therefore did not predict runtime. RSS is whole-process memory.",
        "RQ3 Corrected Pareto Trade-offs": "RQ3 Exploratory Pareto Trade-offs",
        "Figure 6 visualizes the corrected exploratory frontier":
            "Under the evaluated objectives, Figure 6 identifies MobileNetV3 and ShuffleNetV2 as non-dominated artifacts. MobileNetV3 has the lowest RTF (0.0146). ShuffleNetV2 combines diagnostic-subset EER = 0.0476, the highest mean stressed F1 (0.8133), and RTF = 0.0208; it dominates the other four artifacts in this diagnostic analysis. The frontier is exploratory because robustness uses 100 recordings and runtime one desktop host; it does not identify a universal optimum.",
        "Evidence is limited by heterogeneous training":
            "Evidence is limited by heterogeneous training and calibration, a fixed 100-recording robustness diagnostic, simulated rather than physical replay, and runtime measurements on one desktop CPU. The dataset lacks metadata needed to establish speaker, source, generator, parent-recording, or cross-dataset independence. No unseen-corpus, edge-device, causal-streaming, or multi-seed evaluation was conducted; RSS represents whole-process memory, EfficientNet-B0 is represented only by its available warm-up checkpoint, and paired scores for sample-level flip analysis are unavailable.",
        "Recommendations are therefore artifact- and platform-specific.":
            "Recommendations are scenario-, artifact-, and platform-specific. Under the evaluated objectives, ShuffleNetV2 provides the strongest measured balance across clean discrimination, diagnostic stressed F1, and RTF. MobileNetV3 is the non-dominated choice when latency is prioritized on the measured host. These findings are not universal architecture rankings or an optimal-model claim.",
        "Overall, LAVA provides an evidence-traceable evaluation":
            "Overall, LAVA provides an evidence-traceable evaluation of six heterogeneous voice anti-spoofing artifacts. Under the evaluated objectives, the exploratory non-dominated set comprises MobileNetV3 and ShuffleNetV2, while low-SNR noise is the principal observed weakness on the fixed diagnostic subset. Future work should retain paired per-recording scores, extend robustness evaluation to the full test set, and evaluate unseen corpora, physical replay, streaming, and target edge hardware [15], [16]. The current evidence supports deployment-oriented offline comparison, but not comprehensive generalization, streaming, or edge-device validation.",
    }
    for prefix, text in replacements.items():
        set_paragraph_text(find_paragraph(doc, prefix), text)

    # Keep the reviewer pass within the existing page budget using a standard compact
    # bibliography treatment; body text, figures, metrics, and limitations are unchanged.
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith("[") and "]" in paragraph.text[:6]:
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = Pt(8.5)
            for run in paragraph.runs:
                run.font.size = Pt(8)

    # Remove template-generated empty lines between the References heading and
    # the first bibliography item; they otherwise force the final entry alone
    # onto an extra page in Word.
    references_heading = find_paragraph(doc, "References")
    sibling = references_heading._p.getnext()
    while sibling is not None and sibling.tag == qn("w:p") and not "".join(sibling.itertext()).strip():
        next_sibling = sibling.getnext()
        sibling.getparent().remove(sibling)
        sibling = next_sibling

    # Remove the template's final empty body paragraph, which Word paginates as
    # a blank thirteenth page after the compact bibliography.
    body = doc._element.body
    for child in list(body)[::-1]:
        if child.tag == qn("w:sectPr"):
            continue
        if child.tag == qn("w:p") and not "".join(child.itertext()).strip():
            body.remove(child)
            continue
        break

    doc.core_properties.subject = (
        "Reviewer-aligned revision: Related Work, evidence-bounded robustness, EfficientNet warm-up provenance, "
        "and exploratory Pareto interpretation"
    )
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
