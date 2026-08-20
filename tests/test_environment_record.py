from tools.write_environment import build_environment_record

REQUIRED_KEYS = {
    "schema",
    "generated_utc",
    "python",
    "platform",
    "packages",
    "binaries",
    "deviations",
    "record_hash",
}

def test_record_has_every_required_key():
    record = build_environment_record()
    assert REQUIRED_KEYS <= set(record)

def test_record_pins_the_packages_the_report_cites():
    record = build_environment_record()
    for name in ("numpy", "torch", "reportlab", "opencv-python-headless"):
        assert name in record["packages"], f"{name} missing from environment record"

def test_record_names_ffmpeg_and_ffprobe():
    record = build_environment_record()
    assert "ffmpeg" in record["binaries"]
    assert "ffprobe" in record["binaries"]

def test_python_deviation_is_declared():
    record = build_environment_record()
    joined = " ".join(record["deviations"]).lower()
    assert "3.11" in joined and "3.12" in joined

def test_record_hash_is_stable_across_two_builds():
    first = build_environment_record()
    second = build_environment_record()
    assert first["record_hash"] == second["record_hash"]
