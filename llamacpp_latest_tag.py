"""Print the tag of the newest official llama.cpp release (for get_llamacpp_latest.py)."""
import json
import urllib.request

req = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                             headers={"Accept": "application/vnd.github+json", "User-Agent": "exact-spec-amd"})
print(json.loads(urllib.request.urlopen(req, timeout=60).read())["tag_name"])
