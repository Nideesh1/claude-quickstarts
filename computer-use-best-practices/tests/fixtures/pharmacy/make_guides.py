"""Render the pinned reference-image "store policy" guides as PNGs.

These are the guides the pinning experiment pins. Each one encodes per-step
*overrides* — values the model would NOT choose on its own (or would actively
get wrong) — so that an arm WITHOUT the guide in context diverges from target.
The task text deliberately omits these values; the guide is the only source.

Reproducible: rebuild with `python tests/fixtures/pharmacy/make_guides.py`.
"""

import functools
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "refs"

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#16202b;background:#fff;width:760px}
.card{border:2px solid #c0392b;border-radius:12px;overflow:hidden}
.hd{background:#0f2a47;color:#fff;padding:14px 20px}
.hd .t{font-size:13px;letter-spacing:.5px;color:#7fb3f2;font-weight:700;text-transform:uppercase}
.hd .h{font-size:21px;font-weight:700;margin-top:2px}
.hd .s{font-size:12px;opacity:.85;margin-top:3px}
.warn{background:#fdecea;color:#c0392b;font-weight:700;font-size:13px;padding:9px 20px;
  border-bottom:1px solid #f3c4bf;text-transform:uppercase;letter-spacing:.3px}
.body{padding:8px 20px 18px}
.rule{display:flex;gap:14px;padding:14px 0;border-bottom:1px solid #e3e9f0}
.rule:last-child{border-bottom:0}
.badge{flex:0 0 auto;background:#1763b6;color:#fff;font-weight:700;font-size:12px;border-radius:6px;
  padding:5px 10px;height:fit-content;white-space:nowrap}
.txt .field{font-size:12px;color:#5b6b7b;font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.txt .val{font-size:17px;font-weight:700;color:#0f2a47;margin:2px 0 3px}
.txt .val .hi{background:#fff3bf;padding:1px 7px;border-radius:5px;border:1px solid #f0dca8}
.txt .why{font-size:13px;color:#5b6b7b}
.foot{background:#f3f7fc;color:#1763b6;font-size:12px;font-weight:600;padding:10px 20px;
  border-top:1px solid #d7dee7}
"""


def _card(title_tag, heading, subtitle, rules):
    rows = ""
    for badge, field, val, why in rules:
        rows += f"""
        <div class="rule">
          <div class="badge">{badge}</div>
          <div class="txt">
            <div class="field">{field}</div>
            <div class="val">{val}</div>
            <div class="why">{why}</div>
          </div>
        </div>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_CSS}</style></head>
    <body><div class="card">
      <div class="hd">
        <div class="t">{title_tag}</div>
        <div class="h">{heading}</div>
        <div class="s">Bay Area Community Pharmacy #4471 · RxIntake Pro · {subtitle}</div>
      </div>
      <div class="warn">⚠ Store policy — overrides default judgement. Apply exactly as written.</div>
      <div class="body">{rows}</div>
      <div class="foot">Keep this policy in view for the whole intake — it applies at the step shown, even later in the record.</div>
    </div></body></html>"""


GUIDES = {
    "guide-intake.png": _card(
        "Store Policy — Guide 1 of 3",
        "Intake: Patient, Allergies, Insurance",
        "Steps 1–3",
        [
            ("STEP 1", "Pregnancy status (pt_preg)",
             'Set to <span class="hi">Unknown</span>',
             "Policy: record Unknown for every female patient unless she explicitly states otherwise. Do not leave N/A."),
            ("STEP 2", "Allergy severity — ALL rows",
             'Grade every row <span class="hi">Moderate</span>',
             "At intake, ALL allergy rows are entered as Moderate. The PharmD re-grades on review. Never enter Mild or Severe at intake — even for anaphylaxis."),
            ("STEP 3", "Cardholder relationship + person code",
             'Relationship <span class="hi">Spouse</span> · Person code <span class="hi">03</span>',
             "This patient is a spouse dependent on the cardholder's plan. Do not use Self."),
        ],
    ),
    "guide-prescription.png": _card(
        "Store Policy — Guide 2 of 3",
        "Dispensing: DAW & Substitution",
        "Step 5 — Prescription",
        [
            ("STEP 5", "DAW code (drug_daw)",
             'Set to <span class="hi">1 — Substitution not allowed (prescriber)</span>',
             "The prescriber marked this brand medically necessary. Do not leave 0."),
            ("STEP 5", "Generic substitution permitted (sub_ok)",
             'UNCHECK this box',
             "Dispense brand only. The checkbox is checked by default — you must uncheck it."),
        ],
    ),
    "guide-review.png": _card(
        "Store Policy — Guide 3 of 3",
        "Pharmacist Review",
        "Step 6 — Dispense & Review",
        [
            ("STEP 6", "Counseling offered (counsel)",
             'Select <span class="hi">Declined</span>',
             "Per the intake note the patient declined counseling at pickup. Do not select Accepted."),
            ("REMINDER", "To submit the record",
             'Interaction check = Yes · tick the attestation box',
             "The form will not submit until both radios are set and the attestation checkbox is ticked."),
        ],
    ),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT.parent / "_guide_html"
    tmp.mkdir(exist_ok=True)
    for name, html in GUIDES.items():
        (tmp / name.replace(".png", ".html")).write_text(html)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp))
    httpd = socketserver.TCPServer(("127.0.0.1", 8755), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 760, "height": 200}, device_scale_factor=2)
        for name in GUIDES:
            pg.goto(f"http://127.0.0.1:8755/{name.replace('.png', '.html')}")
            pg.wait_for_timeout(150)
            card = pg.query_selector(".card")
            card.screenshot(path=str(OUT / name))
            print(f"wrote {OUT / name}")
        b.close()
    httpd.shutdown()
    for f in tmp.glob("*.html"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
