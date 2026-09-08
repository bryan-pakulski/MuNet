"""Fetch the pinned, checksum-verified upstream source used by parity tests."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]

def main():
    manifest=json.loads((ROOT/'examples/rtdetr/tests/rtdetr-reference.json').read_text())
    output=ROOT/'artifacts/rtdetr-reference';output.mkdir(parents=True,exist_ok=True)
    for name,entry in manifest['files'].items():
        path=output/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']: continue
        url=f"https://raw.githubusercontent.com/lyuwenyu/RT-DETR/{manifest['commit']}/{entry['path']}"
        with urllib.request.urlopen(url,timeout=60) as response: data=response.read(1024*1024)
        if hashlib.sha256(data).hexdigest()!=entry['sha256']: raise ValueError(f'reference hash mismatch: {name}')
        path.write_bytes(data)
    print(f'Verified RT-DETR reference {manifest["commit"]} in {output}')

if __name__=='__main__':main()
