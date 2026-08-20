"""Small ReportLab report writer for fallback examinations."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from peri.core.canon import sha256_file
from peri.core.ledger import Ledger


def write_report(findings: dict, out_path: str | Path | None = None) -> Path:
    evidence_dir = Path("evidence") / findings["evidence_id"]
    path = Path(out_path) if out_path else evidence_dir / "report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    _width, height = A4

    def new_page(title):
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(54, height - 54, title)
        return height - 82

    # Page 1: Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(54, height - 54, "Peri-Peri Fries Examination Summary")
    y = height - 82
    c.setFont("Helvetica", 9)
    rows = [
        ("Evidence ID", findings["evidence_id"]),
        ("Examiner", findings.get("examiner", "unattributed")),
        ("Generated (UTC)", findings.get("generated_utc", "")),
        ("Original SHA-256", findings["exhibit"]["original_sha256"]),
        ("Findings hash", findings["findings_hash"]),
        ("Outcome", findings["decision"]["outcome"]),
        ("Log10 LR Total", str(findings["decision"].get("log10lr_total", 0))),
        ("Reason codes", ", ".join(findings["decision"]["reason_codes"])),
    ]
    for label, value in rows:
        c.drawString(54, y, f"{label}: {value}"[:120])
        y -= 16

    # Page 2: Exhibit
    y = new_page("Exhibit Details")
    c.setFont("Helvetica", 9)
    c.drawString(54, y, f"Original Hash: {findings['exhibit']['original_sha256']}")
    y -= 16
    c.drawString(54, y, f"Working Copy Hash: {findings['exhibit']['working_sha256']}")
    y -= 16
    c.drawString(54, y, f"Container: {findings['exhibit']['container']}")
    y -= 16

    # Page 3: Methods & Calibration
    y = new_page("Methods and Calibration")
    c.setFont("Helvetica", 9)
    c.drawString(54, y, "Methods: Noiseprint (2019), TruFor (CVPR 2023), DiCoME (ICML 2026), DTRA (ICMR 2026), GenD (WACV 2026), NTIRE 2026")
    y -= 16
    c.drawString(54, y, "Calibration Corpus: " + (findings.get("calibration", {}).get("corpus_id", "Unknown") if findings.get("calibration", {}).get("available") else "Missing Calibration"))
    y -= 16
    c.drawString(54, y, "Validated Domain: " + str(findings.get("calibration", {}).get("validated_domain", "None")))
    y -= 16

    # Page 4: Findings
    y = new_page("Findings")
    c.setFont("Helvetica", 9)
    c.drawString(54, y, "Hp: " + findings["propositions"]["Hp"])
    y -= 16
    c.drawString(54, y, "Hd: " + findings["propositions"]["Hd"])
    y -= 16
    c.drawString(54, y, "Decision: " + findings["decision"]["sentence"])
    y -= 16

    # Page 5: Localised Findings
    y = new_page("Localised Findings")
    c.setFont("Helvetica", 9)
    loc = findings.get("localisation", {})
    c.drawString(54, y, f"Suspect frames count: {len(loc.get('top_suspect_frames', []))}")
    y -= 16

    # Page 6: Integrity / Ledger
    y = new_page("Integrity and Chain of Custody")
    c.setFont("Helvetica", 9)
    ledger_path = evidence_dir / "ledger.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            c.drawString(54, y, line[:100])
            y -= 14
            if y < 50:
                y = new_page("Integrity and Chain of Custody (cont.)")
                c.setFont("Helvetica", 9)

    # Page 7: Reproducibility
    y = new_page("Reproducibility")
    c.setFont("Helvetica", 9)
    c.drawString(54, y, f"Manifest hash: {findings.get('manifest_hash', '')}")
    y -= 16
    c.drawString(54, y, "Models: " + str(findings.get("models", {})))
    y -= 16

    # Page 8: Limitations
    y = new_page("Limitations")
    c.setFont("Helvetica", 9)
    lines = [
        "Automated detection is probabilistic. Absence of detected manipulation does not establish authenticity.",
        "Findings are conditional on the declared validated domain; exhibits outside that domain are reported as inconclusive.",
        "This report is forensic decision support prepared to assist an examiner.",
        "It does not itself constitute the certificate under Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023,",
        "and does not determine admissibility or evidentiary weight, which are matters for the Court."
    ]
    for line in lines:
        c.drawString(54, y, line)
        y -= 14

    # Page 9: Section 63(4) Part-B
    y = new_page("Section 63(4) Part-B Draft")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(54, y, "DRAFT - REQUIRES EXPERT REVIEW AND SIGNATURE")
    y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(54, y, "This material serves as a draft input for the human expert.")
    y -= 16

    c.save()
    Ledger(evidence_dir / "ledger.jsonl").append(
        "REPORT_GENERATED",
        findings["evidence_id"],
        {"report_sha256": sha256_file(path), "filename": path.name},
    )
    return path
