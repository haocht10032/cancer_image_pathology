"""Verify the publication export without importing torch or executing experiments."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import re

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "provenance/export_manifest.json").read_text())
expanded = {e["destination"][:-3]: e["decoded_sha256"] for e in manifest
            if e["destination"].endswith(".csv.gz")}
problems = []
for entry in manifest:
    path = root / entry["destination"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["published_sha256"]:
        problems.append(f"Export hash mismatch: {entry['destination']}")
secret = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{25,})")
private = re.compile(r"/(?:Users|users|oscar/home|home)/[A-Za-z0-9_.-]+/")
notebooks = 0
for path in root.rglob("*"):
    if not path.is_file() or any(p in {".git", "__pycache__"} for p in path.parts):
        continue
    if path.suffix in {".pt", ".pth", ".safetensors", ".docx", ".tif", ".tiff"}:
        problems.append(f"Excluded file type present: {path.relative_to(root)}")
    relative = path.relative_to(root).as_posix()
    if relative in expanded:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expanded[relative]:
            problems.append(f"Modified expanded CSV: {relative}")
    if path.stat().st_size > 50_000_000 and relative not in expanded:
        problems.append(f"Oversized Git file: {path.relative_to(root)}")
    data = gzip.decompress(path.read_bytes()) if path.name.endswith(".csv.gz") else path.read_bytes()
    try: text = data.decode("utf-8")
    except UnicodeDecodeError: continue
    if secret.search(text) or private.search(text):
        problems.append(f"Credential-like string or private path: {path.relative_to(root)}")
    if path.suffix == ".py":
        ast.parse(text)
    if path.suffix == ".ipynb":
        nb = json.loads(text); notebooks += 1
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert not cell.get("outputs") and cell.get("execution_count") is None
                ast.parse("".join(cell["source"]))
if problems:
    raise SystemExit("\n".join(problems))
print(f"PASS: {len(manifest)} export hashes; {notebooks} clean notebooks; Python syntax; credential/path/size checks.")
print("This is a packaging check, not a license approval or a GPU reproduction test.")
