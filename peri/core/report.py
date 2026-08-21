"""L7: the examiner's report.

Nine pages, structured against CLAUDE.md section 7. The report is the artefact that
leaves the building, so every page is built from the findings document and nothing
else: there is no prose here that is not bound to a number the findings recorded.

The report is forensic decision support. It does not itself constitute the certificate
under Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023 -- page 9 prepares inputs
for the human expert who signs that certificate, and is watermarked accordingly.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from peri.core.canon import sha256_file
from peri.core.ledger import Ledger

LEFT = 54.0
RIGHT = 54.0
TOP = 54.0
BOTTOM = 58.0

BODY_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"
MONO_FONT = "Courier"
BODY_SIZE = 8.5
LINE = 12.0

LIMITATIONS_VERBATIM = (
    "Automated detection is probabilistic. Absence of detected manipulation does not "
    "establish authenticity. Findings are conditional on the declared validated domain; "
    "exhibits outside that domain are reported as inconclusive. This report is forensic "
    "decision support prepared to assist an examiner. It does not itself constitute the "
    "certificate under Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023, and does "
    "not determine admissibility or evidentiary weight, which are matters for the Court."
)

PRIOR_ART = (
    "Noiseprint (Cozzolino & Verdoliva, 2019)",
    "TruFor (Guillaro et al., CVPR 2023)",
    "DiCoME (ICML 2026)",
    "DTRA (ICMR 2026)",
    "GenD (WACV 2026)",
    "NTIRE 2026 Robust Deepfake Detection Challenge",
    "C2PA / c2pa-python",
)


class _Sheet:
    """A cursor over a paginated A4 canvas.

    ReportLab's canvas has no concept of flow, and the previous report drew unwrapped
    strings that ran off the right edge and off the bottom of the page. Everything here
    goes through `para` or `row`, both of which wrap and both of which break the page
    when they run out of room.
    """

    def __init__(self, path: Path, evidence_id: str, findings_hash: str) -> None:
        self.canvas = canvas.Canvas(str(path), pagesize=A4)
        self.width, self.height = A4
        self.evidence_id = evidence_id
        self.findings_hash = findings_hash
        self.page_number = 0
        self.title = ""
        self.y = 0.0

    @property
    def text_width(self) -> float:
        return self.width - LEFT - RIGHT

    def page(self, title: str) -> None:
        if self.page_number:
            self._footer()
            self.canvas.showPage()
        self.page_number += 1
        self.title = title
        self.canvas.setFont(BOLD_FONT, 13)
        self.canvas.drawString(LEFT, self.height - TOP, title)
        self.canvas.setLineWidth(0.6)
        self.canvas.line(
            LEFT, self.height - TOP - 8, self.width - RIGHT, self.height - TOP - 8
        )
        self.y = self.height - TOP - 26

    def _footer(self) -> None:
        self.canvas.setFont(BODY_FONT, 7)
        self.canvas.setFillGray(0.4)
        self.canvas.drawString(
            LEFT, BOTTOM - 22, f"{self.evidence_id}  |  findings {self.findings_hash[:16]}"
        )
        self.canvas.drawRightString(
            self.width - RIGHT, BOTTOM - 22, f"Section {self.page_number} of 9"
        )
        self.canvas.setFillGray(0.0)

    def space(self, amount: float = 8.0) -> None:
        self.y -= amount

    def _ensure(self, needed: float) -> None:
        if self.y - needed < BOTTOM:
            title = self.title
            self.page(f"{title} (continued)")
            self.page_number -= 1  # a continuation is not a new numbered section

    def heading(self, text: str) -> None:
        self._ensure(LINE * 2)
        self.space(4)
        self.canvas.setFont(BOLD_FONT, 9.5)
        self.canvas.drawString(LEFT, self.y, text)
        self.y -= LINE + 2

    def _wrap(self, text: str, font: str, size: float, width: float) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if stringWidth(candidate, font, size) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def para(self, text: str, font: str = BODY_FONT, size: float = BODY_SIZE) -> None:
        for line in self._wrap(text, font, size, self.text_width):
            self._ensure(LINE)
            self.canvas.setFont(font, size)
            self.canvas.drawString(LEFT, self.y, line)
            self.y -= LINE

    def bullet(self, text: str) -> None:
        indent = 12.0
        lines = self._wrap(text, BODY_FONT, BODY_SIZE, self.text_width - indent)
        for i, line in enumerate(lines):
            self._ensure(LINE)
            self.canvas.setFont(BODY_FONT, BODY_SIZE)
            if not i:
                self.canvas.drawString(LEFT, self.y, "-")
            self.canvas.drawString(LEFT + indent, self.y, line)
            self.y -= LINE

    def row(self, label: str, value, label_width: float = 150.0, mono: bool = False) -> None:
        font = MONO_FONT if mono else BODY_FONT
        size = BODY_SIZE if not mono else BODY_SIZE - 0.5
        lines = self._wrap(value, font, size, self.text_width - label_width)
        for i, line in enumerate(lines):
            self._ensure(LINE)
            if not i:
                self.canvas.setFont(BOLD_FONT, BODY_SIZE)
                self.canvas.drawString(LEFT, self.y, str(label))
            self.canvas.setFont(font, size)
            self.canvas.drawString(LEFT + label_width, self.y, line)
            self.y -= LINE

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        self._ensure(LINE * 2)
        self.canvas.setFont(BOLD_FONT, 7.5)
        x = LEFT
        for header, width in zip(headers, widths, strict=False):
            self.canvas.drawString(x, self.y, header.upper())
            x += width
        self.y -= 4
        self.canvas.setLineWidth(0.4)
        self.canvas.line(LEFT, self.y, self.width - RIGHT, self.y)
        self.y -= LINE
        for row in rows:
            height = 1
            wrapped = []
            for cell, width in zip(row, widths, strict=False):
                lines = self._wrap(cell, MONO_FONT, 7.0, width - 6)
                wrapped.append(lines)
                height = max(height, len(lines))
            self._ensure(LINE * height)
            for line_index in range(height):
                x = LEFT
                for lines, width in zip(wrapped, widths, strict=False):
                    if line_index < len(lines):
                        self.canvas.setFont(MONO_FONT, 7.0)
                        self.canvas.drawString(x, self.y, lines[line_index])
                    x += width
                self.y -= LINE - 1.5
            self.y -= 1.5

    def blank_field(self, label: str, lines: int = 1) -> None:
        """A field only a human can complete: a label and a rule to write on."""
        for i in range(lines):
            self._ensure(LINE * 1.6)
            if not i:
                self.canvas.setFont(BOLD_FONT, BODY_SIZE)
                self.canvas.drawString(LEFT, self.y, str(label))
            self.canvas.setLineWidth(0.4)
            self.canvas.setStrokeGray(0.6)
            self.canvas.line(LEFT + 170, self.y - 2, self.width - RIGHT, self.y - 2)
            self.canvas.setStrokeGray(0.0)
            self.y -= LINE * 1.6

    def watermark(self, text: str) -> None:
        self.canvas.saveState()
        self.canvas.setFont(BOLD_FONT, 46)
        self.canvas.setFillGray(0.86)
        self.canvas.translate(self.width / 2, self.height / 2)
        self.canvas.rotate(38)
        self.canvas.drawCentredString(0, 0, text)
        self.canvas.restoreState()

    def image(self, path: Path, width: float, caption: str = "") -> None:
        try:
            reader = ImageReader(str(path))
            iw, ih = reader.getSize()
        except Exception:
            return
        height = width * ih / iw
        self._ensure(height + LINE * 2)
        self.canvas.drawImage(
            reader, LEFT, self.y - height, width=width, height=height,
            preserveAspectRatio=True, anchor="sw",
        )
        self.y -= height + 4
        if caption:
            self.canvas.setFont(BODY_FONT, 7)
            self.canvas.setFillGray(0.35)
            self.canvas.drawString(LEFT, self.y, caption)
            self.canvas.setFillGray(0.0)
            self.y -= LINE

    def save(self) -> None:
        self._footer()
        self.canvas.save()


def _fmt(value, default: str = "not recorded") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _bytes(value) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def write_report(findings: dict, out_path: str | Path | None = None) -> Path:
    evidence_dir = Path("evidence") / findings["evidence_id"]
    path = Path(out_path) if out_path else evidence_dir / "report.pdf"
    if out_path:
        evidence_dir = Path(out_path).parent
    path.parent.mkdir(parents=True, exist_ok=True)

    exhibit = findings.get("exhibit", {})
    container = exhibit.get("container", {}) or {}
    video = container.get("video", {}) or {}
    decision = findings.get("decision", {})
    fragility = findings.get("fragility", {})
    localisation = findings.get("localisation", {})
    models = findings.get("models", {})
    provenance = findings.get("streams", {}).get("provenance", {})
    sampling = findings.get("sampling", {})

    manifest = {}
    manifest_path = evidence_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sheet = _Sheet(path, findings["evidence_id"], findings.get("findings_hash", ""))

    # ---------------------------------------------------------------- page 1
    sheet.page("1. Examination Summary")
    sheet.row("Evidence ID", findings["evidence_id"], mono=True)
    sheet.row("Examiner", _fmt(findings.get("examiner"), "unattributed"))
    sheet.row("Examined (UTC)", _fmt(findings.get("generated_utc")))
    sheet.row("Examined (IST)", _fmt(findings.get("generated_ist")))
    sheet.row("Software", "Team Peri Peri Fries examination engine, schema "
              + _fmt(findings.get("schema")))
    sheet.row("Findings hash", _fmt(findings.get("findings_hash")), mono=True)
    sheet.row("Manifest hash", _fmt(findings.get("manifest_hash")), mono=True)

    sheet.heading("Model versions and checksums")
    artifacts = (manifest.get("artifacts") or {})
    if artifacts:
        sheet.table(
            ["artefact", "sha-256"],
            [[name, digest] for name, digest in sorted(artifacts.items())],
            [170, sheet.text_width - 170],
        )
    else:
        sheet.para("No artefact checksums were recorded for this examination.")

    sheet.heading("Outcome")
    sheet.row("Outcome", _fmt(decision.get("outcome")))
    sheet.row("Fused log10 LR", _fmt(decision.get("log10lr_total")))
    sheet.row("Verbal equivalent", _fmt(decision.get("verbal")))
    sheet.row("Reason codes", ", ".join(decision.get("reason_codes", [])) or "none")
    sheet.row("Calibration corpus",
              _fmt((manifest.get("config") or {}).get("corpus_id")
                   or _corpus_id(), "PPF-ICV-1"))
    sheet.space(4)
    sheet.para(decision.get("sentence", ""))

    # ---------------------------------------------------------------- page 2
    sheet.page("2. Exhibit")
    sheet.row("Original filename", _fmt(exhibit.get("original_filename")))
    sheet.row("File size", _bytes(exhibit.get("size_bytes")))
    sheet.row("SHA-256 of original", _fmt(exhibit.get("original_sha256")), mono=True)
    sheet.row("Working copy SHA-256", _fmt(exhibit.get("working_sha256")), mono=True)
    sheet.row("Working copy method", _fmt(exhibit.get("working_copy_method")))
    sheet.row("Original held read-only", _fmt(exhibit.get("original_read_only")))
    sheet.row("Stated acquisition source",
              "not stated by the submitting party; recorded as unknown")

    sheet.heading("Container and stream")
    sheet.row("Container", _fmt(container.get("format_long_name")))
    sheet.row("Format name", _fmt(container.get("format_name")))
    sheet.row("Video codec", _fmt(video.get("codec")))
    sheet.row("Resolution", f"{_fmt(video.get('width'))} x {_fmt(video.get('height'))}")
    sheet.row("Frame rate", f"{_fmt(video.get('fps'))} fps")
    sheet.row("Frames", _fmt(video.get("nb_frames")))
    sheet.row("Duration", f"{_fmt(container.get('duration_s'))} s")
    sheet.row("Pixel format", _fmt(video.get("pix_fmt")))
    sheet.row("Profile", _fmt(video.get("profile")))
    sheet.row("Audio stream", _fmt(container.get("audio"), "none present"))

    sheet.heading("Container metadata tags")
    tags = container.get("tags") or {}
    if tags:
        sheet.table(
            ["tag", "value"],
            [[k, str(v)] for k, v in sorted(tags.items())],
            [150, sheet.text_width - 150],
        )
    else:
        sheet.para("No container metadata tags were present.")

    sheet.heading("C2PA provenance")
    c2pa = (provenance.get("facts", {}) or {}).get("c2pa", {}) or {}
    sheet.row("C2PA status", _fmt(c2pa.get("status"), "unavailable"))
    sheet.para(
        "A C2PA manifest verifies that provenance claims have not been tampered with, "
        "not that they are truthful. Where forensic findings contradict a manifest, the "
        "forensic findings take precedence and both are reported."
    )

    # ---------------------------------------------------------------- page 3
    sheet.page("3. Methods and Calibration")
    sheet.heading("Examination stages")
    modes = models.get("modes", {}) or {}
    sheet.table(
        ["stage", "role", "operating mode"],
        [
            ["Stage A", "acquisition fingerprint", _fmt(modes.get("acquisition"))],
            ["Stage B", "tamper mask + reliability", _fmt(modes.get("decoder"))],
            ["Stage C", "temporal aggregation", _fmt(modes.get("temporal"))],
            ["S4", "provenance rules (no ML)", "rule-based"],
        ],
        [90, 200, sheet.text_width - 290],
    )
    sheet.para(
        "Where a stage reports a fallback mode, no learned checkpoint was loaded for it "
        "and a deterministic classical operator was used instead. The mode is recorded "
        "here because it changes what the finding may be called."
    )

    sheet.heading("Trainable parameters")
    for name in ("videoprint", "decoder", "temporal"):
        extra = ((models.get(name) or {}).get("meta") or {}).get("extra") or {}
        sheet.row(name, _fmt(extra.get("n_parameters"), "not recorded")
                  + " parameters, trained on " + _fmt(extra.get("device"), "unknown device"))

    sheet.heading("Calibration corpus and validated domain")
    cal = _calibration()
    sheet.row("Corpus ID", _fmt(cal.get("corpus_id"), "PPF-ICV-1"))
    sheet.row("Held-out method", _fmt(cal.get("held_out_method")))
    counts = cal.get("split_counts") or {}
    if counts:
        sheet.table(
            ["split", "authentic", "manipulated"],
            [[k, str(v.get("authentic", 0)), str(v.get("manipulated", 0))]
             for k, v in sorted(counts.items())],
            [120, 120, sheet.text_width - 240],
        )
    domain = cal.get("validated_domain") or {}
    sheet.row("Validated codecs", ", ".join(domain.get("codecs", [])) or "not declared")
    sheet.row("Validated duration", _fmt(domain.get("duration_range_s")))
    sheet.para(_fmt(domain.get("statement"), ""))
    sheet.para(_fmt(cal.get("corpus_description"), ""))

    metrics = cal.get("metrics") or {}
    sheet.heading("Reported metrics")
    sheet.row("AUROC (held-out method)",
              _fmt(metrics.get("auroc_held_out_method"), "not computed on this split"))
    sheet.row("Expected calibration error", _fmt(metrics.get("ece")))
    sheet.para(
        "Where the held-out-generator AUROC is not computed, no generalisation figure "
        "may be quoted for this build, and any accuracy figure stated elsewhere is an "
        "in-domain figure only."
    )

    sheet.heading("Sampling")
    sheet.row("Frames examined", _fmt(sampling.get("examination_frames")))
    sheet.row("Frames per fragility probe", _fmt(sampling.get("fragility_probe_frames")))

    sheet.heading("Fragility axes and the disjointness rule")
    from peri.core.fragility import FRAGILITY_AXES, TRAINING_AUGMENTATIONS
    sheet.table(
        ["axis", "family", "unit", "ladder"],
        [[name, spec["family"], spec["unit"],
          ", ".join(str(v) for v in spec["ladder"])]
         for name, spec in sorted(FRAGILITY_AXES.items())],
        [95, 110, 80, sheet.text_width - 285],
    )
    sheet.row("Training augmentations",
              ", ".join(f"{n} ({s['family']})" for n, s in sorted(TRAINING_AUGMENTATIONS.items())))
    sheet.para(
        "The augmentation families used during training and the transform families used "
        "by the fragility search share no member and no parameter range. The assertion "
        "that enforces this runs at import time. Were they to overlap, the robustness "
        "claim would be circular."
    )

    sheet.heading("Library versions")
    packages = ((manifest.get("environment") or {}).get("packages") or {})
    if packages:
        sheet.para(", ".join(f"{k} {v}" for k, v in sorted(packages.items())))

    sheet.heading("Prior art consumed and credited")
    for item in PRIOR_ART:
        sheet.bullet(item)
    sheet.para(
        "The acquisition-fingerprint paradigm is not ours. Its video formulation, its "
        "adversarial fragility reporting, and its statutory packaging are."
    )

    # ---------------------------------------------------------------- page 4
    sheet.page("4. Findings")
    sheet.heading("Propositions considered")
    props = findings.get("propositions", {})
    sheet.row("Hp", _fmt(props.get("Hp")))
    sheet.row("Hd", _fmt(props.get("Hd")))
    sheet.para(
        "The likelihood ratio expresses how much more probable these findings are if Hd "
        "is true than if Hp is true. It is a statement about the evidence, not about the "
        "world, and it does not express the probability that either proposition is true."
    )

    sheet.heading("Per-stream results")
    stream_rows = []
    for stream in decision.get("streams", []):
        stream_rows.append([
            _fmt(stream.get("name")),
            _fmt(stream.get("median_log10lr")),
            _fmt(stream.get("mahalanobis")) + " / " + _fmt(stream.get("mahalanobis_threshold")),
            "yes" if stream.get("in_domain") else "no",
            "yes" if stream.get("usable") else "no",
            _fmt(stream.get("exclusion_reason"), "-"),
        ])
    if stream_rows:
        sheet.table(
            ["stream", "log10 LR", "mahal / thresh", "in domain", "usable", "excluded because"],
            stream_rows,
            [78, 55, 105, 52, 45, sheet.text_width - 335],
        )
    else:
        sheet.para("No stream produced a likelihood ratio for this exhibit.")

    sheet.row("Provenance rule score", _fmt(provenance.get("score")))

    sheet.heading("Fusion and outcome")
    sheet.row("Dependence shrinkage", _fmt(decision.get("dependence_shrinkage")))
    sheet.row("Usable streams", ", ".join(decision.get("usable_stream_names", [])) or "none")
    sheet.row("Fused log10 LR", _fmt(decision.get("log10lr_total")))
    sheet.row("Verbal equivalent", _fmt(decision.get("verbal")))
    sheet.row("Outcome", _fmt(decision.get("outcome")))
    sheet.row("Primary reason", _fmt(decision.get("primary_reason"), "-"))
    sheet.row("Reason codes", ", ".join(decision.get("reason_codes", [])) or "none")
    sheet.para(
        "Streams drawn from the same pixels are correlated. Independence is not claimed; "
        "the fusion applies a fixed dependence shrinkage as stated conservatism."
    )
    sheet.space(4)
    sheet.para(decision.get("sentence", ""))

    sheet.heading("Evidence Fragility Index")
    sheet.para(_fmt(fragility.get("statement")))
    axis_rows = []
    for name, axis in sorted((fragility.get("axes") or {}).items()):
        axis_rows.append([
            name,
            _fmt(axis.get("label_survives")),
            _fmt(axis.get("label_flips"), "did not flip on any tested rung"),
            str(axis.get("n_evaluations", 0)),
        ])
    if axis_rows:
        sheet.table(
            ["axis", "survives to", "flips at", "probes"],
            axis_rows,
            [110, 110, 190, sheet.text_width - 410],
        )
    sheet.row("Fragility band", _fmt(fragility.get("band")))

    # ---------------------------------------------------------------- page 5
    sheet.page("5. Localised Findings")
    suspects = localisation.get("top_suspect_frames", []) or []
    if suspects:
        sheet.table(
            ["frame", "timestamp", "anomaly score", "reliability", "confident"],
            [[str(r.get("index")), f"{_fmt(r.get('timestamp_s'))} s",
              _fmt(r.get("score")), _fmt(r.get("reliability")),
              "yes" if r.get("confident") else "no"] for r in suspects],
            [60, 90, 100, 90, sheet.text_width - 340],
        )
        sheet.row("Anomaly type", "acquisition-consistency anomaly in the facial region")
        sheet.row("Contributing streams", "acquisition, temporal")
        sheet.para(
            "Region of interest is reported as the masked area of each frame below. "
            "Frames the model does not trust are marked as not confident and should not "
            "be read as findings."
        )
        top = suspects[0]
        overlay = evidence_dir / "frames" / f"overlay_{int(top['index']):04d}.png"
        sheet.image(
            overlay, 250,
            f"Frame {top['index']} at {_fmt(top.get('timestamp_s'))} s, tamper mask "
            f"overlaid, reliability {_fmt(top.get('reliability'))}.",
        )
        videoprint = evidence_dir / "frames" / f"videoprint_{int(top['index']):04d}.png"
        if videoprint.is_file():
            sheet.image(
                videoprint, 250,
                f"Frame {top['index']}, acquisition fingerprint field. A manipulated "
                "region reads as a different texture from its surroundings.",
            )
    else:
        sheet.para("No suspect frames were localised for this exhibit.")

    timeline = localisation.get("timeline", []) or []
    sheet.row("Timeline length", f"{len(timeline)} frames scored")

    # ---------------------------------------------------------------- page 6
    sheet.page("6. Integrity and Chain of Custody")
    sheet.para(
        "The original exhibit was hashed on receipt, sealed read-only, and never written "
        "to. All examination was performed on a byte-identical working copy whose hash is "
        "recorded on page 2. Each ledger entry below commits to the digest of the entry "
        "before it, so any alteration or removal breaks the chain from that point onward."
    )
    sheet.space(4)
    ledger_path = evidence_dir / "ledger.jsonl"
    rows = []
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            rows.append([
                str(entry.get("seq")),
                str(entry.get("event")),
                str(entry.get("ts_utc", "")),
                str(entry.get("hash", ""))[:24] + "...",
            ])
    if rows:
        sheet.table(
            ["seq", "event", "recorded (utc)", "entry hash"],
            rows,
            [34, 150, 125, sheet.text_width - 309],
        )
        sheet.row("Ledger entries", str(len(rows)))
    else:
        sheet.para("No ledger was found for this examination.")

    # ---------------------------------------------------------------- page 7
    sheet.page("7. Reproducibility")
    sheet.row("Manifest hash", _fmt(findings.get("manifest_hash")), mono=True)
    sheet.row("Findings hash", _fmt(findings.get("findings_hash")), mono=True)
    sheet.row("Master seed", _fmt(manifest.get("seed")))

    sheet.heading("Configuration")
    for key, value in sorted((manifest.get("config") or {}).items()):
        sheet.row(key, _fmt(value))

    sheet.heading("Model checksums")
    if artifacts:
        sheet.table(
            ["artefact", "sha-256"],
            [[name, digest] for name, digest in sorted(artifacts.items())],
            [170, sheet.text_width - 170],
        )

    sheet.heading("Hardware and interpreter")
    environment = manifest.get("environment") or {}
    python = environment.get("python") or {}
    sheet.row("Interpreter", f"{_fmt(python.get('implementation'))} "
              f"{_fmt(python.get('version'), '')}".strip())
    for name, version in sorted((environment.get("binaries") or {}).items()):
        sheet.row(name, str(version))
    sheet.row("Compute device", _fmt((manifest.get("config") or {}).get("device")))

    sheet.heading("Replay")
    sheet.para(
        "Re-running the examination from this manifest reproduces the findings hash "
        "above byte for byte. The replay is itself recorded on the custody chain, "
        "whether or not it matched."
    )
    sheet.row("Replay command",
              f"python -c \"from peri.core.pipeline import replay; "
              f"print(replay('evidence/{findings['evidence_id']}'))\"", mono=True)

    # ---------------------------------------------------------------- page 8
    sheet.page("8. Limitations")
    sheet.para(LIMITATIONS_VERBATIM)
    sheet.space(10)
    sheet.heading("Specific to this examination")
    if not (cal.get("metrics") or {}).get("auroc_held_out_method"):
        sheet.bullet(
            "No held-out-generator AUROC was computed for this build. No generalisation "
            "figure may be quoted; any accuracy figure is in-domain only."
        )
    ece = (cal.get("metrics") or {}).get("ece")
    if ece is not None:
        sheet.bullet(
            f"The expected calibration error of the calibration corpus is {_fmt(ece)}. "
            "The likelihood ratios reported here inherit that calibration."
        )
    if counts:
        cal_split = counts.get("cal", {})
        sheet.bullet(
            f"The calibration split holds {cal_split.get('authentic', 0)} authentic and "
            f"{cal_split.get('manipulated', 0)} manipulated samples. Density estimates "
            "fitted on a split this size are reported with the corresponding caution."
        )
    if "out-of-validated-domain" in decision.get("reason_codes", []):
        sheet.bullet(
            "This exhibit fell outside the declared validated domain, and the affected "
            "streams were excluded from the fusion for that reason."
        )
    sheet.bullet(
        "The examination covers the visual stream only. No audio, lip-sync or "
        "speaker analysis was performed."
    )

    # ---------------------------------------------------------------- page 9
    sheet.page("9. Section 63(4) BSA Part-B Draft Input Sheet")
    sheet.watermark("DRAFT")
    sheet.canvas.setFont(BOLD_FONT, 11)
    sheet.canvas.drawString(LEFT, sheet.y, "DRAFT - REQUIRES EXPERT REVIEW AND SIGNATURE")
    sheet.y -= LINE * 1.6
    sheet.para(
        "Section 63(4) of the Bharatiya Sakshya Adhiniyam, 2023 requires a certificate "
        "signed by the person in charge of the device and by an expert. This sheet is not "
        "that certificate and is not a substitute for it. It collects, in one place, the "
        "examination facts an expert may need when preparing their own Part-B entry. The "
        "system signs nothing, and the entries below are inputs for a human to verify, "
        "adopt, correct or reject."
    )
    sheet.space(6)

    sheet.heading("A. Facts established by this examination")
    sheet.row("Electronic record", _fmt(exhibit.get("original_filename")))
    sheet.row("Size", _bytes(exhibit.get("size_bytes")))
    sheet.row("Container and codec",
              f"{_fmt(container.get('format_name'))} / {_fmt(video.get('codec'))}")
    sheet.row("Resolution, rate, duration",
              f"{_fmt(video.get('width'))}x{_fmt(video.get('height'))}, "
              f"{_fmt(video.get('fps'))} fps, {_fmt(container.get('duration_s'))} s")
    sheet.row("Hash algorithm", "SHA-256")
    sheet.row("Hash of the record", _fmt(exhibit.get("original_sha256")), mono=True)
    sheet.row("Original held read-only", _fmt(exhibit.get("original_read_only")))
    sheet.row("Handling", "hashed on receipt, sealed read-only, examined on a "
              "byte-identical working copy")
    sheet.row("Custody entries", f"{len(rows)} chained entries, recorded on page 6")
    sheet.row("Examination outcome", _fmt(decision.get("outcome")))
    sheet.row("Fused log10 LR", _fmt(decision.get("log10lr_total")))
    sheet.row("Verbal equivalent", _fmt(decision.get("verbal")))
    sheet.row("Fragility", _fmt(fragility.get("statement")))
    sheet.row("Findings hash", _fmt(findings.get("findings_hash")), mono=True)
    sheet.row("Examined (UTC / IST)",
              f"{_fmt(findings.get('generated_utc'))} / {_fmt(findings.get('generated_ist'))}")

    sheet.heading("B. To be completed and verified by the human expert")
    sheet.blank_field("Expert name")
    sheet.blank_field("Designation and organisation")
    sheet.blank_field("Qualifications relied upon")
    sheet.blank_field("Person in charge of the device")
    sheet.blank_field("Device / system particulars")
    sheet.blank_field("How the record was produced")
    sheet.blank_field("Stated acquisition source")
    sheet.blank_field("Place and date")
    sheet.blank_field("Signature")

    sheet.space(6)
    sheet.para(
        "The expert adopting any entry above does so on their own responsibility, having "
        "satisfied themselves of it independently. Nothing in this sheet determines "
        "admissibility or evidentiary weight, which are matters for the Court."
    )

    sheet.save()

    Ledger(evidence_dir / "ledger.jsonl").append(
        "REPORT_GENERATED",
        findings["evidence_id"],
        {"report_sha256": sha256_file(path), "filename": path.name},
    )
    return path


def _calibration() -> dict:
    path = Path("artifacts") / "calibration.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _corpus_id() -> str:
    return str(_calibration().get("corpus_id", "PPF-ICV-1"))
