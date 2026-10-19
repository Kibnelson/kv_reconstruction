"""Fetch the pinned public pretrained checkpoint used by the smoke test."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, pathlib, time, urllib.request

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
FILES = ["config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
         "merges.txt", "model.safetensors"]

def download(destination):
    dest = pathlib.Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    def one(name):
        target = dest / name
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        if not target.exists():
            for attempt in range(3):
                temporary = target.with_suffix(target.suffix + ".part")
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                        while chunk := response.read(1024 * 1024):
                            out.write(chunk)
                    os.replace(temporary, target)
                    break
                except Exception:
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(1)
        digest = hashlib.sha256()
        with target.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        print(f"downloaded {name}: {target.stat().st_size} bytes", flush=True)
        return {"name":name,"bytes":target.stat().st_size,"sha256":digest.hexdigest(),"url":url}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(one, FILES))
    manifest = {"model_id":MODEL_ID,"revision":REVISION,"files":files,
                "note":"Public Apache-2.0 checkpoint; weights are not bundled with the smoke-test results."}
    (dest/"download_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--destination",required=True)
    args=parser.parse_args()
    download(args.destination)

