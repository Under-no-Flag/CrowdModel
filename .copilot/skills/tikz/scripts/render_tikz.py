#!/usr/bin/env python3
"""Render TikZ code to SVG via tectonic/pdflatex + pdf2svg."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from config import get_host, get_port

CACHE_DIR = Path.home() / ".cache" / "tikz-skill"
RENDERS_DIR = CACHE_DIR / "renders"


def find_tex_engine():
    for engine in ["tectonic", "pdflatex"]:
        if shutil.which(engine):
            return engine
    return None


def find_svg_converter():
    if shutil.which("pdf2svg"):
        return "pdf2svg"
    if shutil.which("pdftocairo"):
        return "pdftocairo"
    return None


def wrap_tikz(tikz_code):
    """Wrap TikZ code in a standalone document if not already a full document."""
    if "\\documentclass" in tikz_code:
        return tikz_code

    PREAMBLE_PREFIXES = (
        "\\usepackage", "\\usetikzlibrary", "\\usepgfplotslibrary",
        "\\definecolor", "\\tikzset", "\\pgfplotsset",
        "\\newcommand", "\\renewcommand", "\\def\\",
        "\\PassOptionsToPackage", "\\tikzstyle",
    )
    preamble, body = [], []
    in_body = False
    for line in tikz_code.strip().splitlines():
        stripped = line.strip()
        if not in_body and any(stripped.startswith(p) for p in PREAMBLE_PREFIXES):
            preamble.append(line)
        else:
            in_body = True
            body.append(line)

    return f"""\\documentclass[border=2pt]{{standalone}}
\\usepackage{{tikz}}
{chr(10).join(preamble)}
\\begin{{document}}
{chr(10).join(body)}
\\end{{document}}
"""


def render(tex_file_path):
    """Render a .tex file to SVG. Returns dict with url or error."""
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    engine = find_tex_engine()
    if not engine:
        return {"error": "No TeX engine found. Install tectonic or a TeX distribution with pdflatex."}
    svg_converter = find_svg_converter()
    if not svg_converter:
        return {"error": "No PDF-to-SVG converter found. Install pdf2svg or Poppler's pdftocairo."}

    tex_path = Path(tex_file_path).resolve()
    tikz_code = tex_path.read_text()
    tex_content = wrap_tikz(tikz_code)

    # Name: stem + timestamp
    stem = tex_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_name = f"{stem}_{ts}"

    with tempfile.TemporaryDirectory(prefix="tikz-") as tmpdir:
        tmp_tex = Path(tmpdir) / "input.tex"
        tmp_pdf = Path(tmpdir) / "input.pdf"
        svg_out = RENDERS_DIR / f"{out_name}.svg"
        pdf_out = RENDERS_DIR / f"{out_name}.pdf"

        tmp_tex.write_text(tex_content)

        if engine == "tectonic":
            result = subprocess.run(
                ["tectonic", str(tmp_tex)],
                capture_output=True, text=True, cwd=tmpdir,
            )
        else:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", tmpdir, str(tmp_tex)],
                capture_output=True, text=True, cwd=tmpdir,
            )

        if result.returncode != 0:
            return {"error": f"TeX compilation failed:\n{result.stdout}\n{result.stderr}"}

        if not tmp_pdf.exists():
            return {"error": "PDF not generated. Check TeX code."}

        if svg_converter == "pdf2svg":
            svg_cmd = ["pdf2svg", str(tmp_pdf), str(svg_out)]
        else:
            svg_cmd = ["pdftocairo", "-svg", str(tmp_pdf), str(svg_out)]

        result = subprocess.run(svg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"error": f"{svg_converter} failed:\n{result.stderr}"}

        shutil.copy2(tmp_pdf, pdf_out)

        # Convert PDF to PNG for visual verification
        png_out = RENDERS_DIR / f"{out_name}.png"
        subprocess.run(
            ["pdftocairo", "-png", "-r", "200", "-singlefile", str(tmp_pdf), str(png_out.with_suffix(""))],
            capture_output=True, text=True,
        )

        host = get_host()
        port = get_port()
        url = f"http://{host}:{port}/{out_name}.svg"

        return {"url": url, "png": str(png_out)}


def main():
    parser = argparse.ArgumentParser(description="Render TikZ code to SVG")
    parser.add_argument("tex_file", help="Path to .tex file")
    args = parser.parse_args()

    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"ERROR: File not found: {args.tex_file}", file=sys.stderr)
        sys.exit(1)

    # Ensure server is running
    ensure_script = Path(__file__).parent / "ensure_server.py"
    srv = subprocess.run([sys.executable, str(ensure_script)], capture_output=True, text=True)
    if srv.returncode != 0:
        print(f"ERROR: Server failed to start: {srv.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    result = render(args.tex_file)

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)
    else:
        print(result["url"])
        print(result["png"])


if __name__ == "__main__":
    main()
