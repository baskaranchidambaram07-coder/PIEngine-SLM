"""Generate the attachment test fixtures under sample_docs/attachments/.

Every file is synthetic. The "pii-*" files carry only well-known placeholder
values — AWS's documented example key, the 4111… test card, the 123-45-6789
SSN from the SSA's own examples, example.com addresses — so the fixtures can
live in a public repo while still tripping every detector they are meant to.

Run:  venv\\Scripts\\python scripts\\make_attachment_samples.py
Needs reportlab (text PDFs), PyMuPDF (image-only PDF) and Pillow (images).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_docs" / "attachments"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "C:/Windows/Fonts/arial.ttf"

CLEAN_INVOICE_TXT = """INVOICE INV-2026-0912
Date: 12 September 2026        Due: 30 September 2026
Supplier: Nandini Office Supplies Pvt Ltd, Bengaluru
Bill to: Acme Corp — Programme Office, purchase order PO-77120

Item                              Qty     Unit       Amount
A4 copier paper (box of 5)         12   INR 1,150   INR 13,800
Whiteboard markers (pack of 10)     6   INR   420   INR  2,520
Laser toner CF287A                  2   INR 6,900   INR 13,800

Subtotal                                             INR 30,120
GST 18%                                              INR  5,422
Total due                                            INR 35,542
Total in USD at 0.0562                               $1,999.00

Payment terms: 15 days net. Reference 4500012345 on your remittance.
"""

CLEAN_NOTES_MD = """# Programme sync — 10 September 2026

## Attendees
Delivery lead, PMO, platform architect, QA lead.

## Decisions
- The Phase 2 rollout date moves from 22 September to **6 October 2026** so UAT can finish.
- QA will run the regression suite twice before go-live, not once.
- The mobile client stays on the current SDK for this release.

## Risks
- The payment gateway vendor has not confirmed the sandbox window. Owner: PMO.
- Two contractors roll off on 30 September; knowledge transfer starts Monday.

## Actions
1. PMO to circulate the revised plan by Friday.
2. Architect to publish the integration checklist.
3. QA lead to book the second regression run.
"""

CLEAN_BRIEF_TEXT = [
    "Product brief: Field Inspection Assistant",
    "",
    "Purpose. Give inspectors a fully offline assistant on a rugged tablet so that",
    "checklists, prior reports and equipment manuals are available without a network.",
    "",
    "Pilot. Three pilot sites are planned for Q4 2026: Pune, Chennai and Kochi. Each",
    "site gets six tablets and a two-week onboarding. Success means a 20% reduction",
    "in report turnaround time and no inspection blocked by connectivity.",
    "",
    "Scope. The assistant answers from the shipped knowledge base only. It does not",
    "submit reports, does not talk to the ERP, and keeps all data on the device.",
    "",
    "Timeline. Bundle authoring in September, device provisioning in October, pilot",
    "review in the first week of December.",
]

SCANNED_POLICY_TEXT = [
    "REMOTE WORK POLICY  (v3, effective 1 August 2026)",
    "",
    "1. Eligibility. Employees past probation may work remotely up to",
    "   three days a week with line-manager approval.",
    "2. Notice. Changes to a standing remote-work pattern require",
    "   30 days written notice to the line manager and HR.",
    "3. Equipment. The company provides one laptop and one headset.",
    "   Home internet costs are not reimbursed.",
    "4. Security. Company data stays on company devices. Personal",
    "   cloud storage is not permitted for work files.",
    "5. Review. This policy is reviewed every twelve months.",
]

RECEIPT_LINES = [
    "NANDINI CAFE - MG ROAD",
    "Bengaluru 560001",
    "Table 7        11 Sep 2026 13:42",
    "--------------------------------",
    "Masala dosa        2 x 9.50   19.00",
    "Filter coffee      2 x 3.25    6.50",
    "Rava kesari        1 x 5.00    5.00",
    "--------------------------------",
    "Subtotal                      30.50",
    "Service 10%                    3.05",
    "GST 18%                        9.20",
    "TOTAL                         42.75",
    "--------------------------------",
    "Thank you! Visit again.",
]

PII_CONTACTS_TXT = """Vendor onboarding form (draft)

Primary contact: Priya Sharma
Email: priya.sharma@example.com
Mobile: +91 98765 43210
Backup contact email: ops-desk@example.org

Please keep this sheet internal.
"""

PII_SECRETS_MD = """# Deployment notes — staging

Bucket sync uses the CI role.

    export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
    export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

The staging database password is Hunter2!Staging until rotation.
api_key = 3f9a8b7c6d5e4f3a2b1c0d9e
"""

PII_CARD_PDF_TEXT = [
    "Expense reimbursement — travel",
    "",
    "Employee ID E-10442",
    "Corporate card used: 4111 1111 1111 1111 (exp 12/28)",
    "Tax reference (US): 123-45-6789",
    "Hotel, two nights, Pune: INR 9,800",
    "Taxi to airport: INR 640",
    "Approved by line manager.",
]

PII_BADGE_LINES = [
    "VISITOR WIFI",
    "Network: ACME-GUEST",
    "Password: Wel(ome2Acme!",
    "Support: helpdesk@example.com",
]


def render_text_image(lines: list[str], width: int = 1100, line_h: int = 44,
                      size: int = 30, mono: bool = False) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFont

    font_path = "C:/Windows/Fonts/consola.ttf" if mono else FONT
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("RGB", (width, 60 + line_h * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((40, 30 + i * line_h), ln, fill="black", font=font)
    return img


def write_text_pdf(path: Path, lines: list[str]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y = h - 72
    c.setFont("Helvetica", 11.5)
    for ln in lines:
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 11.5)
            y = h - 72
        c.drawString(72, y, ln)
        y -= 16
    c.showPage()
    c.save()


def write_scanned_pdf(path: Path, lines: list[str]) -> None:
    """An image-only PDF: no text layer at all, so extraction must OCR."""
    import pymupdf

    img = render_text_image(lines, width=1400, line_h=52, size=34, mono=True)
    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=80)   # a scan, not a screenshot
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(30, 40, 565, 40 + 535 * img.height / img.width), stream=buf.getvalue())
    doc.save(str(path))
    doc.close()


def main() -> None:
    (OUT / "clean-invoice.txt").write_text(CLEAN_INVOICE_TXT, encoding="utf-8")
    (OUT / "clean-notes.md").write_text(CLEAN_NOTES_MD, encoding="utf-8")
    write_text_pdf(OUT / "clean-brief.pdf", CLEAN_BRIEF_TEXT)
    write_scanned_pdf(OUT / "scanned-policy.pdf", SCANNED_POLICY_TEXT)
    render_text_image(RECEIPT_LINES, width=760, line_h=40, size=26, mono=True).save(
        OUT / "clean-receipt.jpg", quality=92)

    (OUT / "pii-contacts.txt").write_text(PII_CONTACTS_TXT, encoding="utf-8")
    (OUT / "pii-secrets.md").write_text(PII_SECRETS_MD, encoding="utf-8")
    write_text_pdf(OUT / "pii-card.pdf", PII_CARD_PDF_TEXT)
    render_text_image(PII_BADGE_LINES, width=900, line_h=50, size=34).save(
        OUT / "pii-badge.jpg", quality=92)

    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size:>8,} B  {p.name}")


if __name__ == "__main__":
    sys.exit(main())
