"""Export seven-page static PDFs from the validated English slide PNGs.

Use a Python environment containing reportlab, pypdf and Pillow:
    python tools/export_slide_pdf.py
PowerPoint charts and tables remain editable in the companion PPTX files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "_build" / "slides"


def export_pdf(number: int) -> None:
    build = BUILD / f"tt{number}"
    receipt = json.loads((build / "validation.json").read_text(encoding="utf-8"))
    deck = ROOT / "slides" / f"THUAT_TOAN_{number}" / f"THUAT_TOAN_{number}.pptx"
    required_checks = ("packageIntegrity", "presentationLayout", "nativeQuantitativeCharts")
    if (any(receipt.get(check, {}).get("exitCode") != 0 for check in required_checks)
            or not receipt.get("firstPartyImport", {}).get("passed")
            or hashlib.sha256(deck.read_bytes()).hexdigest() != receipt.get("finalSha256")):
        raise ValueError(f"{build}: the current PPTX must match a successful validation receipt")
    images = [build / f"slide-{index}.png" for index in range(1, 8)]
    for image in images:
        with Image.open(image) as source:
            if source.size != (1280, 720):
                raise ValueError(f"Unexpected slide dimensions: {image}")
    destination = ROOT / "slides" / f"THUAT_TOAN_{number}" / f"THUAT_TOAN_{number}.pdf"
    candidate = build / "candidate.pdf"
    pdf = canvas.Canvas(str(candidate), pagesize=(960, 540))
    pdf.setTitle(f"Algorithm {number}")
    pdf.setSubject("Speech and silence segmentation")
    for image in images:
        pdf.drawImage(str(image), 0, 0, width=960, height=540)
        pdf.showPage()
    pdf.save()
    if len(PdfReader(candidate).pages) != 7:
        raise ValueError(f"{candidate}: expected seven PDF pages")
    if destination.exists():
        previous = BUILD / "previous" / datetime.now().strftime("%Y%m%d-%H%M%S")
        previous.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, previous / destination.name)
    shutil.copy2(candidate, destination)
    sheet = Image.new("RGB", (1280, 4 * 385), "#d9e1ea")
    for index, image in enumerate(images):
        with Image.open(image) as source:
            sheet.paste(source.convert("RGB").resize((640, 360)),
                        (index % 2 * 640, index // 2 * 385))
        ImageDraw.Draw(sheet).text((index % 2 * 640 + 8, index // 2 * 385 + 362),
                                  f"Algorithm {number} / slide {index + 1}", fill="black")
    sheet.save(build / "contact.png")
    print(f"Algorithm {number}: seven PDF pages exported to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", type=int, choices=(1, 2, 3))
    arguments = parser.parse_args()
    for number in [arguments.algorithm] if arguments.algorithm else range(1, 4):
        export_pdf(number)
