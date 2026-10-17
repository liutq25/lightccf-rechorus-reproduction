"""Record immutable source URLs and SHA256 hashes for this read-only audit."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
commit = json.loads((root / "official_commit.json").read_text(encoding="utf-8-sig"))["sha"]
code_paths = {
    "official_LightCCF.py": "LightCCF.py",
    "official_LightCCF.sh": "LightCCF.sh",
    "official_main.py": "main.py",
    **{f"official_{name}.py": f"utility/{name}.py" for name in
       ["data_loader", "losses", "parser", "tester", "tools", "trainer"]},
}
urls = {
    "LightCCF_2504.10113.pdf": "https://arxiv.org/pdf/2504.10113v1",
    "LightCCF_2504.10113.html": "https://arxiv.org/html/2504.10113v1",
    **{local: f"https://raw.githubusercontent.com/ZzYUuuu/LightCCF/{commit}/{remote}"
       for local, remote in code_paths.items()},
}
manifest = {
    "retrieved_date": "2026-10-16",
    "timezone": "Asia/Shanghai",
    "official_commit": commit,
    "sources": [
        {"local_path": name, "url": url,
         "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
         "bytes": (root / name).stat().st_size}
        for name, url in urls.items()
    ],
}
(root / "paper_source_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {len(manifest['sources'])} source files at commit {commit}")
