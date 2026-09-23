"""Write the AiZynthFinder asset manifest to /app/models/manifest.json.

Called during Docker build (after model download + MD5 verification).
SHA-256 values are PLACEHOLDER because integrity was already verified
via MD5 during the Zenodo download step.
"""
import json
import pathlib

manifest = {
    "_comment": [
        "AiZynthFinder models downloaded from Zenodo record 11430881.",
        "Policy and templates are baked into the image; stock is mounted at runtime.",
        "MD5 checksums were verified at download time (see Dockerfile RUN step).",
        "SHA-256 is PLACEHOLDER — file existence verified by manifest loader.",
    ],
    "policy": {
        "name": "uspto_rop_onnx",
        "version": "zenodo_11430881",
        "file": "policy/uspto_model.onnx",
        "sha256": "PLACEHOLDER",
        "source_url": "https://zenodo.org/records/11430881",
        "license": "MIT",
    },
    "templates": {
        "name": "uspto_templates",
        "version": "zenodo_11430881",
        "file": "templates/uspto_templates.csv.gz",
        "sha256": "PLACEHOLDER",
        "source_url": "https://zenodo.org/records/11430881",
        "license": "MIT",
    },
    "stock": [
        {
            "key": "default",
            "name": "zinc_in_stock",
            "version": "zenodo_11430881",
            "file": "stock/zinc_stock.hdf5",
            "sha256": "PLACEHOLDER",
            "source_url": "https://zenodo.org/records/11430881",
            "license": "CC0",
            "note": "Mount at runtime: -v /data/aizynthfinder/stock:/app/models/stock:ro",
        }
    ],
}

out = pathlib.Path("/app/models/manifest.json")
out.write_text(json.dumps(manifest, indent=2))
print(f"manifest.json written to {out}")
