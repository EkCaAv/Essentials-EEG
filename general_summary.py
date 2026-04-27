import os
import sys
import argparse
import pdfkit
from pypdf import PdfReader, PdfWriter

DEFAULT_SUBJECTS = ["05", "09", "14", "16", "20", "22", "23"]

def parse_args():
    parser = argparse.ArgumentParser(description="Convierte reportes HTML por sujeto a PDF y los consolida")
    parser.add_argument("--results_root", type=str, default="results_pediatric_study")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--wkhtml_path", type=str, default=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    parser.add_argument("--merged_output", type=str, default="CHB_6_10_reportes_unidos.pdf")
    parser.add_argument("--single_output_dir", type=str, default="reporting/resultsbyPDF")
    return parser.parse_args()


args = parse_args()

WKHTML = args.wkhtml_path
if not os.path.exists(WKHTML):
    print("ERROR: No encuentro wkhtmltopdf en:", WKHTML)
    sys.exit(1)

config = pdfkit.configuration(wkhtmltopdf=WKHTML)

options = {
    "page-size": "A4",
    "margin-top": "10mm",
    "margin-right": "10mm",
    "margin-bottom": "10mm",
    "margin-left": "10mm",
    "enable-local-file-access": "",
    "javascript-delay": "3000",
    "no-stop-slow-scripts": "",
}

os.makedirs(args.single_output_dir, exist_ok=True)

pdfs = []
for s in args.subjects:
    html_file = os.path.abspath(f"{args.results_root}/chb{s}/reports/chb{s}_report.html")
    if not os.path.exists(html_file):
        print("ERROR: No existe el HTML:", html_file)
        sys.exit(1)

    out_pdf = os.path.abspath(os.path.join(args.single_output_dir, f"chb{s}_report.pdf"))
    print(f"Generando PDF: {os.path.basename(out_pdf)}")
    pdfkit.from_file(html_file, out_pdf, configuration=config, options=options)
    pdfs.append(out_pdf)

merged_out = os.path.abspath(args.merged_output)
writer = PdfWriter()

for p in pdfs:
    print("Uniendo:", os.path.basename(p))
    reader = PdfReader(p)
    for page in reader.pages:
        writer.add_page(page)

with open(merged_out, "wb") as f:
    writer.write(f)

print("OK:", merged_out)
