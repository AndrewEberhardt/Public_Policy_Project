#!/bin/sh
# Export the policy memo to PDF: run memo_policy.ipynb, drop code and working notes, print with headless Chrome (A4).
set -e
cd "$(dirname "$0")"
jupyter nbconvert memo_policy.ipynb --execute --inplace
jupyter nbconvert memo_policy.ipynb --to html --no-input --no-prompt \
  --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' \
  --output memo_policy_print
python3 - <<'PY'
css = """<style>
@page { size: A4; margin: 1.2cm 1.4cm; }
body, .jp-Notebook, .jp-RenderedHTMLCommon { font-family: "Helvetica Neue", Arial, sans-serif !important;
  font-size: 12pt !important; line-height: 1.2 !important; color: #111; }
.jp-Notebook { padding: 0 !important; }
.jp-Cell { padding: 0 !important; margin: 0 !important; }
.jp-RenderedHTMLCommon h1 { font-size: 16pt !important; margin: 0 0 6px 0 !important; }
.jp-RenderedHTMLCommon h2 { font-size: 13.5pt !important; margin: 8px 0 3px 0 !important; }
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li { margin: 0 0 4px 0 !important; }
.jp-RenderedHTMLCommon ul, .jp-RenderedHTMLCommon ol { margin: 0 0 4px 0 !important; padding-left: 18px !important; }
.jp-RenderedHTMLCommon hr { margin: 4px 0 !important; }
.jp-RenderedHTMLCommon table { font-size: 8pt !important; margin: 2px 0 !important; width: 100%; }
.jp-RenderedHTMLCommon th, .jp-RenderedHTMLCommon td { padding: 1px 4px !important; }
.memo-note { font-size: 7.8pt !important; font-style: italic; color: #333; margin: 2px 0 6px 0 !important; }
.jp-RenderedHTMLCommon td:first-child { min-width: 11.5em; }
.jp-OutputArea-output { break-inside: avoid; }
.memo-fig img, .jp-OutputArea-output img { max-width: 64% !important; display: block; margin: 0 auto; }
.memo-fig { break-inside: avoid; }
.jp-OutputArea-output, .jp-OutputArea-child { margin: 0 !important; padding: 0 !important; }
table, img { page-break-inside: avoid; }
</style>"""
p = "memo_policy_print.html"
s = open(p, encoding="utf-8").read().replace("</head>", css + "</head>", 1)
open(p, "w", encoding="utf-8").write(s)
PY
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/memo_policy.pdf" "file://$PWD/memo_policy_print.html" 2>/dev/null
rm memo_policy_print.html
echo "memo_policy.pdf written"
