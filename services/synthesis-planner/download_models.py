"""Download AiZynthFinder policy model and templates from Zenodo (record 11430881).

Called during Docker build. Verifies MD5 checksums after each download.
"""
import hashlib
import pathlib
import urllib.request

MODELS = [
    {
        "url": "https://zenodo.org/records/11430881/files/uspto_model.onnx?download=1",
        "dest": "/app/models/policy/uspto_model.onnx",
        "md5": "b466a436b6a8eceaba8c14ef5e2a88c8",
    },
    {
        "url": "https://zenodo.org/records/11430881/files/uspto_templates.csv.gz?download=1",
        "dest": "/app/models/templates/uspto_templates.csv.gz",
        "md5": "e8fb29d4dcd54073658000e212b7845a",
    },
]


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


for model in MODELS:
    dest = pathlib.Path(model["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} …", flush=True)
    for attempt in range(1, 4):
        try:
            urllib.request.urlretrieve(model["url"], str(dest))
            break
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}", flush=True)
            if attempt == 3:
                raise
    actual = _md5(str(dest))
    if actual != model["md5"]:
        raise RuntimeError(
            f"MD5 mismatch for {dest.name}: expected {model['md5']}, got {actual}"
        )
    print(f"  OK ({dest.stat().st_size // 1024 // 1024} MB, md5 verified)", flush=True)

print("All models downloaded and verified.")
