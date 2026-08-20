import math
from peri.core import canon

def test_q_rounds_to_six_places():
    assert canon.q(1 / 3) == 0.333333
    assert canon.q(2.0) == 2.0
    assert canon.q(-1e-9) == 0.0 or canon.q(-1e-9) == -0.0

def test_qdeep_walks_nested_structures():
    src = {"b": [1 / 3, {"c": 2 / 3}], "a": 1, "s": "x", "n": None, "t": True}
    out = canon.qdeep(src)
    assert out["b"][0] == 0.333333
    assert out["b"][1]["c"] == 0.666667
    assert out["a"] == 1 and out["s"] == "x" and out["n"] is None and out["t"] is True

def test_canonical_json_is_key_sorted_and_tight():
    assert canon.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

def test_hash_obj_is_insertion_order_independent():
    assert canon.hash_obj({"a": 1, "b": 2}) == canon.hash_obj({"b": 2, "a": 1})

def test_hash_obj_is_float_noise_independent():
    left = {"score": 0.1 + 0.2}
    right = {"score": 0.3}
    assert left["score"] != right["score"]
    assert canon.hash_obj(left) == canon.hash_obj(right)

def test_sha256_hex_known_vector():
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert canon.sha256_hex(b"") == empty

def test_sha256_file_matches_sha256_hex(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"peri")
    assert canon.sha256_file(p) == canon.sha256_hex(b"peri")

def test_utc_and_ist_timestamps_have_expected_suffixes():
    assert canon.utc_now_iso().endswith("Z")
    assert canon.ist_now_iso().endswith("+05:30")

def test_nan_and_inf_are_rejected():
    for bad in (math.nan, math.inf, -math.inf):
        try:
            canon.canonical_json({"x": bad})
        except ValueError:
            continue
        raise AssertionError("canonical_json must reject non-finite floats")

def test_seed_everything_makes_random_reproducible():
    import random
    canon.seed_everything()
    a = [random.random() for _ in range(5)]
    canon.seed_everything()
    b = [random.random() for _ in range(5)]
    assert a == b
