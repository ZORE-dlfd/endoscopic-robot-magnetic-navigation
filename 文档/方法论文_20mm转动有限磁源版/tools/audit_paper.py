from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    pdf = ROOT / "output/pdf/八线圈磁驱动与进退镜视觉协同方法.pdf"
    reader = PdfReader(pdf)
    texts = [page.extract_text() or "" for page in reader.pages]
    source = (ROOT / "source/methodology.tex").read_text("utf-8")
    log = (ROOT / "build/methodology.log").read_text("utf-8", errors="replace")
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", source)
    citations = re.findall(r"\\cite\{([^}]+)\}", source)
    missing_refs = sorted(set(refs)-set(labels))
    missing_citations = sorted(set(citations)-set(bibliography))
    warnings = [line for line in log.splitlines() if any(term in line for term in ["Overfull", "Missing character:", "undefined", "LaTeX Error"])]
    pages = []
    for index, page in enumerate(reader.pages):
        pages.append({"page": index+1, "width_pt": float(page.mediabox.width), "height_pt": float(page.mediabox.height),
                      "text_characters": len(texts[index]), "replacement_characters": texts[index].count("\ufffd")})
    snapshot = json.loads((ROOT / "audit/source_hashes.json").read_text("utf-8"))
    project = ROOT.parents[1]
    drift = [entry["path"] for entry in snapshot if hashlib.sha256((project/entry["path"]).read_bytes()).hexdigest()!=entry["sha256"]]
    poppler = shutil.which("pdftoppm")
    if poppler is None:
        raise RuntimeError("pdftoppm is required for visual QA")
    previews = ROOT / "build/pages"
    previews.mkdir(exist_ok=True)
    process = subprocess.run([poppler, "-r", "115", "-png", str(pdf), str(previews/"page")], capture_output=True, text=True, errors="replace")
    if process.returncode:
        raise RuntimeError(process.stderr)
    images = sorted(previews.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    for start in range(0, len(images), 6):
        sheet = Image.new("RGB", (3*510, 2*755), "#D5DAE0")
        draw = ImageDraw.Draw(sheet)
        for offset, image_path in enumerate(images[start:start+6]):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((490, 706))
            left = (offset%3)*510+(510-image.width)//2
            top = (offset//3)*755+35
            sheet.paste(image,(left,top))
            draw.text(((offset%3)*510+15,(offset//3)*755+7),f"Page {start+offset+1}",font=font,fill="#172A3A")
        sheet.save(ROOT / f"build/contact_{start//6+1:02d}.jpg",quality=93)
    summary = {"pdf": str(pdf), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "bytes": pdf.stat().st_size,
               "page_count": len(reader.pages), "figure_count": source.count(r"\begin{figure}"),
               "numbered_math_environments": len(re.findall(r"\\begin\{(?:equation|align)\}", source)),
               "bibliography_count": len(bibliography), "missing_references": missing_refs,
               "missing_citations": missing_citations, "blocking_log_warnings": warnings,
               "project_source_drift": drift, "pages": pages, "rendered_pages": len(images),
               "visual_review": "Pending human/agent inspection of all contact sheets and selected full-resolution pages",
               "scope": "Document and internal data consistency audit; not scientific, clinical, or accessibility certification"}
    (ROOT / "audit/pdf_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),"utf-8")
    (ROOT / "audit/paper_text.txt").write_text("\n\n".join(f"PAGE {index+1}\n{text}" for index,text in enumerate(texts)),"utf-8")
    print(json.dumps({key:value for key,value in summary.items() if key!="pages"},ensure_ascii=False,indent=2))
    if missing_refs or missing_citations or warnings or drift or len(images)!=len(reader.pages):
        sys.exit(1)


if __name__ == "__main__":
    main()
