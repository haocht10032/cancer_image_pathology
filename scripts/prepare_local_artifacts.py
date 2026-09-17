"""Expand hash-verified compact CSVs without overwriting different local files."""
from pathlib import Path
import gzip
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "provenance/export_manifest.json").read_text())
count = 0
for entry in manifest:
    if not entry["destination"].endswith(".csv.gz"):
        continue
    source = (root / entry["destination"]).resolve()
    target = source.with_suffix("")
    if root not in source.parents or root not in target.parents:
        raise ValueError("Artifact path escapes repository")
    compressed = source.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != entry["published_sha256"]:
        raise ValueError(f"Compressed hash mismatch: {source.relative_to(root)}")
    data = gzip.decompress(compressed)
    if hashlib.sha256(data).hexdigest() != entry["decoded_sha256"]:
        raise ValueError(f"Decoded hash mismatch: {source.relative_to(root)}")
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite modified artifact: {target.relative_to(root)}")
        continue
    target.write_bytes(data)
    count += 1
print(f"Expanded {count} CSVs. Run reporting from the repository root.")
