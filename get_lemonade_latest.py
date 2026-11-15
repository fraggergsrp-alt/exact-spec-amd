"""B6b helper: download the newest Lemonade (AMD) llama.cpp ROCm build for RDNA4 (gfx120X, Windows) into tools/,
verify its SHA-256 (GitHub's asset digest, else the release's checksums.txt), unpack, and record the tag in
tools/lemonade_latest.txt. These builds bundle the ROCm runtime libraries (the official llama.cpp ROCm zip does not,
which made B6 silently run on the CPU). Nothing is installed system-wide. Safe to re-run.
"""
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
API = "https://api.github.com/repos/lemonade-sdk/llamacpp-rocm/releases/latest"


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "memorywall-lab"})
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    TOOLS.mkdir(exist_ok=True)
    rel = json.loads(get(API))
    tag = rel["tag_name"]
    print(f"Lemonade llamacpp-rocm latest: {tag} published {rel.get('published_at')}")
    print((rel.get("body") or "")[:600])
    asset = next((a for a in rel["assets"] if re.search(r"windows.*rocm.*gfx120X.*x64\.zip$", a["name"])), None)
    if asset is None:
        print("ERROR: no Windows gfx120X asset in", [a["name"] for a in rel["assets"]])
        return 1
    dest = TOOLS / f"llama-lemonade-{tag}"
    if not any(dest.rglob("llama-server.exe")):
        want = (asset.get("digest") or "").removeprefix("sha256:").lower()
        if not want:
            sums = next((a for a in rel["assets"] if a["name"] == "checksums.txt"), None)
            if sums:
                txt = get(sums["browser_download_url"]).decode()
                m = re.search(r"([0-9a-fA-F]{64})\s+\*?" + re.escape(asset["name"]), txt)
                want = m.group(1).lower() if m else ""
        z = TOOLS / asset["name"]
        if not z.exists():
            print(f"downloading {asset['name']} ({asset['size'] / 2**20:.0f} MiB)...", flush=True)
            urllib.request.urlretrieve(asset["browser_download_url"], z)
        h = hashlib.sha256()
        with open(z, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        print(f"sha256 expected {want or '(none published)'}\nsha256 got      {got}")
        if not want or got != want:
            print("ERROR: checksum missing or mismatched - not unpacking")
            return 1
        with zipfile.ZipFile(z) as zz:
            zz.extractall(dest)
        print("checksum OK, unpacked")
    exe = next(dest.rglob("llama-server.exe"), None)
    print("llama-server:", exe)
    (TOOLS / "lemonade_latest.txt").write_text(tag, encoding="utf-8")
    return 0 if exe else 1


if __name__ == "__main__":
    sys.exit(main())
