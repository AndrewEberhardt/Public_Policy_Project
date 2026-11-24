#!/bin/sh
# Export the technical report to PDF: run report_technical.ipynb, drop code and working notes, print with headless Chrome (A4).
set -e
cd "$(dirname "$0")"
jupyter nbconvert report_technical.ipynb --execute --inplace
jupyter nbconvert report_technical.ipynb --to html --no-input --no-prompt \
  --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["remove_cell"]' \
  --output report_technical_print
python3 - <<'PY'
css = """<style>
@page { size: A4; margin: 1.3cm 1.5cm; }
body, .jp-Notebook, .jp-RenderedHTMLCommon { font-family: "Helvetica Neue", Arial, sans-serif !important;
  font-size: 10pt !important; line-height: 1.22 !important; color: #111; }
.jp-Notebook { padding: 0 !important; }
.jp-Cell { padding: 0 !important; margin: 0 !important; }
.jp-RenderedHTMLCommon h1 { font-size: 14.5pt !important; margin: 0 0 5px 0 !important; }
.jp-RenderedHTMLCommon h2 { font-size: 12pt !important; margin: 9px 0 3px 0 !important; }
.jp-RenderedHTMLCommon p { margin: 0 0 4px 0 !important; text-align: justify; }
.jp-OutputArea-output, .jp-OutputArea-child { margin: 0 !important; padding: 0 !important; }
.tblock { break-inside: avoid; margin: 4px 0 6px 0; }
.ttitle { font-weight: bold; font-size: 9pt !important; margin: 0 0 2px 0 !important; }
.tnote { font-size: 7.3pt !important; font-style: italic; color: #333; margin: 2px 0 0 0 !important; line-height: 1.15 !important; text-align: left !important; }
table.rtab { border-collapse: collapse; width: 100%; font-size: 7.6pt !important; margin: 0 !important; }
table.rtab th { border-top: 1.2px solid #000; border-bottom: 0.8px solid #000; font-weight: bold; text-align: center !important; padding: 1.5px 3px !important; background: none !important; }
table.rtab td { padding: 1px 3px !important; text-align: center !important; border: none !important; background: none !important; }
table.rtab tr:last-child td { border-bottom: 1.2px solid #000 !important; }
table.rtab td:first-child, table.rtab th:first-child { text-align: left !important; }
table.rtab tbody tr:nth-child(odd) { background: none !important; }
table.reg { font-size: 7.2pt !important; }
table.reg td { padding: 0.3px 3px !important; }
table.small { font-size: 7pt !important; }
table.rtab { table-layout: auto !important; } table.rtab td:first-child { white-space: nowrap; } table.b4 td:first-child { white-space: normal; min-width: 17em; }
table.a1 { font-size: 7pt !important; } table.a1 td:last-child { text-align: left !important; width: 33%; white-space: normal; }
table.a1 th:last-child { width: 33%; } table.a1 th { padding: 1.5px 5px !important; }
.tblock img { max-width: 88% !important; display: block; margin: 0 auto; }
</style>"""
p = "report_technical_print.html"
s = open(p, encoding="utf-8").read().replace("</head>", css + "</head>", 1)
open(p, "w", encoding="utf-8").write(s)
PY
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/report_technical.pdf" "file://$PWD/report_technical_print.html" 2>/dev/null
rm report_technical_print.html
echo "report_technical.pdf written"
