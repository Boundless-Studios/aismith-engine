from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any


def build_report(manifest_path: Path, output_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()

    def retain(relative: str) -> str:
        safe = Path(relative).name
        if safe not in copied:
            shutil.copy2(manifest_path.parent / relative, output_dir / safe)
            copied.add(safe)
        return html.escape(safe, quote=True)

    source = retain(str(manifest["source"]))
    sections: list[str] = []
    for stage in manifest.get("stages", []):
        settings = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in stage.get("settings", {}).items()
        )
        renders = "".join(
            f'<a href="{retain(str(path))}"><img src="{retain(str(path))}" alt="{html.escape(str(stage["name"]))} render"></a>'
            for path in stage.get("renders", [])
        )
        artifacts = "".join(
            f'<a class="artifact" href="{retain(str(path))}" download>{html.escape(Path(str(path)).name)}</a>'
            for path in stage.get("artifacts", [])
        )
        animated = stage.get("animated_glb")
        viewer = ""
        if animated:
            viewer = f'<model-viewer src="{retain(str(animated))}" camera-controls autoplay animation-name="*" shadow-intensity="1"></model-viewer>'
        sections.append(f'''<section><header><div><span>STAGE</span><h2>{html.escape(str(stage["name"]))}</h2></div><b class="{html.escape(str(stage["verdict"]))}">{html.escape(str(stage["verdict"]))}</b></header><div class="grid">{renders}</div>{viewer}<table>{settings}</table><div class="artifacts">{artifacts}</div></section>''')

    document = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(str(manifest["title"]))}</title><script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.1.0/model-viewer.min.js"></script><style>
body{{margin:0;background:#0b0d12;color:#e9eaf0;font:15px Inter,system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:44px 24px}}h1{{font-size:38px;margin:5px 0 28px}}h2{{margin:4px 0}}.eyebrow,header span{{font:11px monospace;letter-spacing:.14em;color:#777f90}}.hero{{display:grid;grid-template-columns:280px 1fr;gap:28px;align-items:center;margin-bottom:30px}}.hero img{{width:100%;max-height:400px;object-fit:contain;background:#11141b;border-radius:14px}}section{{padding:20px;margin:14px 0;border:1px solid #292e3a;border-radius:14px;background:#12151c}}section header{{display:flex;justify-content:space-between;align-items:center}}header b{{padding:6px 10px;border-radius:20px;text-transform:uppercase;font:11px monospace}}header b.accepted{{background:#17352f;color:#91e7d5}}header b.rejected{{background:#3b1d26;color:#ff9caf}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:18px 0}}.grid img{{width:100%;height:210px;object-fit:contain;background:#090b0f;border-radius:8px}}model-viewer{{width:100%;height:520px;background:#090b0f;border-radius:10px}}table{{margin-top:15px;border-collapse:collapse}}th,td{{padding:7px 18px 7px 0;text-align:left;border-bottom:1px solid #242832}}th{{color:#858c9d}}.artifacts{{display:flex;gap:8px;margin-top:14px}}.artifact{{padding:8px 10px;border:1px solid #393f4d;border-radius:6px;color:#d6d9e4;text-decoration:none}}@media(max-width:700px){{.hero{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">REPRODUCIBLE CHARACTER PIPELINE</div><h1>{html.escape(str(manifest["title"]))}</h1><div class="hero"><img src="{source}" alt="Source character"><p>This report retains every reviewed stage. Accepted means visual inspection passed; metadata alone is never sufficient.</p></div>{''.join(sections)}</main></body></html>'''
    output = output_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build_report(args.manifest, args.output))


if __name__ == "__main__":
    main()
