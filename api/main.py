"""FastAPI shell for local examination demos."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from peri.core.errors import PeriError
from peri.core.ledger import verify_ledger
from peri.core.pipeline import examine, replay
from peri.core.report import write_report

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "evidence"
WEB_ROOT = ROOT / "web"
# Uploads land inside evidence/, which .gitignore already excludes, so a demo
# run never litters the repository root with temporary media.
UPLOAD_ROOT = EVIDENCE_ROOT / "_uploads"

app = FastAPI(title="Peri-Peri Fries")
JOBS: dict[str, dict] = {}
# FastAPI resolves this marker at import time; calling File() inside the default
# argument list would re-evaluate it on every request.
UPLOAD_FIELD = File(...)


@app.exception_handler(PeriError)
async def peri_error_handler(_request, exc: PeriError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


def _run_examination(temp_path: Path, job_id: str) -> None:
    try:
        def update_job(msg):
            if "evidence_id" in msg:
                JOBS[job_id]["evidence_id"] = msg["evidence_id"]

        findings = examine(temp_path, evidence_root=EVIDENCE_ROOT, examiner="demo", progress=update_job)
        JOBS[job_id] = {"state": "done", "evidence_id": findings["evidence_id"]}
    except Exception as exc:
        JOBS[job_id] = {"state": "error", "error": str(exc)}
    finally:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)


@app.post("/examine")
async def examine_upload(background_tasks: BackgroundTasks, file: UploadFile = UPLOAD_FIELD):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_ROOT) as handle:
        temp_path = Path(handle.name)
        shutil.copyfileobj(file.file, handle)
    job_id = temp_path.stem
    JOBS[job_id] = {"state": "running", "filename": file.filename}
    background_tasks.add_task(_run_examination, temp_path, job_id)
    return {"job_id": job_id, "state": "running"}


@app.get("/status/{identifier}")
async def status(identifier: str):
    async def events():
        sent = 0
        while True:
            job = JOBS.get(identifier)
            evidence_id = identifier
            if job:
                yield f"event: job\ndata: {json.dumps(job)}\n\n"
                evidence_id = job.get("evidence_id", identifier)
                if job.get("state") == "error":
                    break
            ledger = EVIDENCE_ROOT / evidence_id / "ledger.jsonl"
            if ledger.is_file():
                lines = ledger.read_text(encoding="utf-8").splitlines()
                for line in lines[sent:]:
                    yield f"event: ledger\ndata: {line}\n\n"
                sent = len(lines)
                if any('"event":"FINDINGS_SEALED"' in line for line in lines):
                    break
            await asyncio.sleep(0.75)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/findings/{evidence_id}")
async def findings(evidence_id: str):
    path = EVIDENCE_ROOT / evidence_id / "findings.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="findings not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/report/{evidence_id}")
async def report(evidence_id: str):
    evidence_dir = EVIDENCE_ROOT / evidence_id
    findings_path = evidence_dir / "findings.json"
    if not findings_path.is_file():
        raise HTTPException(status_code=404, detail="findings not found")
    pdf = evidence_dir / "report.pdf"
    if not pdf.is_file():
        write_report(json.loads(findings_path.read_text(encoding="utf-8")), pdf)
    return FileResponse(pdf, media_type="application/pdf", filename="report.pdf")


@app.get("/ledger/{evidence_id}")
async def ledger(evidence_id: str):
    path = EVIDENCE_ROOT / evidence_id / "ledger.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="ledger not found")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {"events": events, "verification": verify_ledger(path).to_dict()}


@app.post("/replay/{evidence_id}")
async def replay_route(evidence_id: str):
    evidence_dir = EVIDENCE_ROOT / evidence_id
    if not evidence_dir.is_dir():
        raise HTTPException(status_code=404, detail="evidence not found")
    return replay(evidence_dir)


@app.get("/frames/{evidence_id}/{name}")
async def frame(evidence_id: str, name: str):
    base = (EVIDENCE_ROOT / evidence_id / "frames").resolve()
    target = (base / name).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="invalid frame path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="frame not found")
    return FileResponse(target)


if WEB_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
