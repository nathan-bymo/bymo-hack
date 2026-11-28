"""Render technical_report.md to the hand-in PDF (BYMO_Report.pdf at the repository root).

Markdown -> HTML (markdown-it, tables on) -> PDF with headless Chrome. Relative links become GitHub links, so they
stay clickable in the PDF; images are embedded from the working tree. Prints the page count.

  .venv/bin/python src/scripts/build_report_pdf.py [--out BYMO_Report.pdf]
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/nathan-bymo/bymo-hack/blob/main/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 14mm 14mm 14mm 14mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 8.6pt; line-height: 1.32; color: #111; }
h1 { font-size: 14pt; margin: 0 0 4pt; }
h2 { font-size: 11pt; margin: 9pt 0 3pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
h3 { font-size: 9.5pt; margin: 7pt 0 2pt; }
p { margin: 3pt 0; }
ul { margin: 2pt 0 3pt 14pt; padding: 0; }
li { margin: 0.6pt 0; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 7.6pt; width: 100%; }
th, td { border: 0.5pt solid #bbb; padding: 1.4pt 3pt; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
code { font-family: Menlo, monospace; font-size: 7.3pt; background: #f4f4f4; padding: 0 1pt; }
pre { background: #f4f4f4; padding: 4pt; font-size: 6.9pt; line-height: 1.25; white-space: pre-wrap; margin: 3pt 0; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 4pt auto; }
a { color: #1a4d8f; text-decoration: none; }
em { color: #333; }
"""


def absolute_links(html):
    def fix(match):
        attr, target = match.group(1), match.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return match.group(0)
        if attr == "src":
            return f'src="{(ROOT / target).as_uri()}"'
        return f'href="{REPO_URL}{target}"'
    return re.sub(r'(href|src)="([^"]+)"', fix, html)


def page_count(pdf):
    data = pdf.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "BYMO_Report.pdf")
    args = parser.parse_args()
    body = MarkdownIt("commonmark").enable("table").render((ROOT / "technical_report.md").read_text(encoding="utf-8"))
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{absolute_links(body)}</body></html>"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={args.out.resolve()}", page.as_uri()], check=True, capture_output=True)
    print(f"{args.out}: {page_count(args.out)} pages")


if __name__ == "__main__":
    main()
