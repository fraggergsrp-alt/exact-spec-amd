"""Print the tag of the newest official llama.cpp build that ships a Windows Vulkan zip (for get_llamacpp_latest.py).
llama.cpp marks its numbered builds (bNNNNN) as pre-releases, so GitHub's "latest release" is not the newest build."""
import json
import urllib.request

req = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30",
                             headers={"Accept": "application/vnd.github+json", "User-Agent": "exact-spec-amd"})
for r in json.loads(urllib.request.urlopen(req, timeout=60).read()):
    t = r["tag_name"]
    if any(a["name"] == f"llama-{t}-bin-win-vulkan-x64.zip" for a in r["assets"]):
        print(t)
        break
