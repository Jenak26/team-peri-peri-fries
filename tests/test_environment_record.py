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
    assert set(record) >= REQUIRED_KEYS

def test_record_pins_the_packages_the_report_cites():
    record = build_environment_record()
    for name in ("numpy", "torch", "reportlab", "opencv-python-headless"):
        assert name in record["packages"], f"{name} missing from environment record"

def test_record_names_ffmpeg_and_ffprobe():
    record = build_environment_record()
    assert "ffmpeg" in record["binaries"]
    assert "ffprobe" in record["binaries"]

def test_python_deviation_names_the_interpreter_actually_running():
    import platform

    record = build_environment_record()
    joined = " ".join(record["deviations"])
    running = platform.python_version()
    if ".".join(running.split(".")[:2]) == "3.11":
        assert "3.11" not in joined, "no deviation to declare on the specified version"
    else:
        assert "3.11" in joined, "the specified version must be named"
        assert running in joined, (
            f"the deviation must name the interpreter that produced the record "
            f"({running}), not a version hard-coded when the file was written"
        )


def test_torch_deviation_matches_the_installed_build():
    record = build_environment_record()
    joined = " ".join(record["deviations"])
    torch_version = record["packages"]["torch"]
    if torch_version != "not-installed":
        assert torch_version in joined
        assert ("CUDA build" in joined) == ("+cu" in torch_version)

def test_record_hash_is_stable_across_two_builds():
    first = build_environment_record()
    second = build_environment_record()
    assert first["record_hash"] == second["record_hash"]
