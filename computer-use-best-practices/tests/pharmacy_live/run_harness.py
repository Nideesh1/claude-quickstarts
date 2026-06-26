"""Live harness: does pinning reference images actually help a real run?

Runs the agent against the local pharmacy fixture in up to three arms and
scores the final form state against the target data:

  baseline  -- task text only, no reference images
  unpinned  -- task + guide images, but pinning OFF (images get pruned)
  pinned    -- task + guide images, pinning ON (images kept all run)

The guide images are annotated how-to screenshots for the tricky steps
(allergies, prescription, review). The prune interval is set small so that in
the `unpinned` arm the guides are stripped well before the late, complex step 6
-- which is exactly where pinning should pay off.

Requires a real ANTHROPIC_API_KEY. Drives a headless Chromium via the repo's
BrowserTool. Usage:

    python tests/pharmacy_live/run_harness.py                 # all 3 arms, opus-4-8
    python tests/pharmacy_live/run_harness.py --arms pinned
    python tests/pharmacy_live/run_harness.py --model claude-sonnet-4-6
"""

import argparse
import functools
import http.server
import os
import socketserver
import sys
import threading
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pharmacy"
REFS = FIXTURE_DIR / "refs"
GUIDES = ["guide-allergies.png", "guide-prescription.png", "guide-review.png"]

# --- config via env, BEFORE importing computer_use (Config reads env at import) ---
os.environ.setdefault("CU_ENABLE_COMPUTER_USE_TOOLS", "false")  # browser only, safe
os.environ.setdefault("CU_ENABLE_BROWSER_USE_TOOLS", "true")
os.environ.setdefault("CU_ENABLE_EDITOR_TOOL", "false")
os.environ.setdefault("CU_IMAGE_PRUNE_STRATEGY", "interval")
os.environ.setdefault("CU_IMAGE_PRUNE_MIN", "2")
# small interval => unpinned guides are pruned within a few turns
os.environ.setdefault("CU_IMAGE_PRUNE_INTERVAL", "4")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from computer_use.__main__ import build_task_content  # noqa: E402
from computer_use.loop import sampling_loop  # noqa: E402
from computer_use.tools import ToolCollection  # noqa: E402
from computer_use.tools.batch import BrowserBatchTool  # noqa: E402
from computer_use.tools.browser import BrowserTool  # noqa: E402
from computer_use.trajectory import Trajectory  # noqa: E402

# ---------------------------------------------------------------------------
# Target data the model must enter. The references are GUIDANCE (how to fill the
# tricky controls), not the data -- the data lives here and in the task text.
# ---------------------------------------------------------------------------
TARGET = {
    "pt_last": "Okonkwo", "pt_first": "Adaeze", "pt_mi": "N",
    "pt_dob": "11/02/1967", "pt_sex": "Female",
    "pt_addr1": "488 Alcatraz Ave", "pt_addr2": "Apt 3B",
    "pt_city": "Oakland", "pt_state": "CA", "pt_zip": "94609",
    "pt_phone": "(510) 555-0173", "pt_email": "adaeze.ok@example.com",
    "nkda": False,
    "ins_plan": "CarelonRx", "ins_member": "CRX774203918", "ins_bin": "020107",
    "ins_pcn": "CRXMEDD", "ins_group": "EBAY2026",
    "rx_prescriber": "Patel, Anita MD", "rx_npi": "1841293756",
    "rx_dea": "BP1234563", "rx_phone": "(510) 555-0190",
    "drug_name": "Alprazolam", "drug_strength": "0.5 mg", "drug_form": "Tablet",
    "drug_ndc": "00603-2128-21", "drug_sched": "CIV",
    "drug_datewritten": "06/20/2026",
    "drug_sig": "Take 1 tablet by mouth twice daily as needed for anxiety",
    "drug_qty": "60", "drug_days": "30", "drug_refills": "2",
    "disp_date": "06/26/2026", "disp_ndc": "00603-2128-21", "disp_qty": "60",
    "disp_rph": "ANR", "counsel": "Accepted", "ddi_done": "Yes", "attest": True,
}
TARGET_ALLERGENS = {"penicillin", "sulfa drugs"}

# Fields whose correctness most depends on the late/tricky steps + guides.
CRITICAL = ["nkda", "drug_form", "drug_ndc", "drug_sched", "rx_dea",
            "drug_daw", "counsel", "ddi_done", "attest"]


def build_task(url: str) -> str:
    return f"""You are a pharmacy technician using RxIntake Pro. Open {url} and fill out the
entire 6-step Pharmacy Dispensing Record, then submit it ("Fill prescription").

Use the attached GUIDE screenshots for HOW to complete the tricky steps
(adding allergy rows, the dropdown fields, and the required radios/attestation
on the final Review step).

Enter EXACTLY this data:

PATIENT: {TARGET['pt_last']}, {TARGET['pt_first']} {TARGET['pt_mi']};
  DOB {TARGET['pt_dob']}; Sex {TARGET['pt_sex']};
  Address {TARGET['pt_addr1']}, {TARGET['pt_addr2']}, {TARGET['pt_city']},
  {TARGET['pt_state']} {TARGET['pt_zip']}; Phone {TARGET['pt_phone']};
  Email {TARGET['pt_email']}.

ALLERGIES (NKDA = NO, the patient HAS allergies):
  1) Penicillin -- Hives, swelling -- Severe
  2) Sulfa drugs -- Rash -- Moderate

INSURANCE: Plan {TARGET['ins_plan']}; Member ID {TARGET['ins_member']};
  BIN {TARGET['ins_bin']}; PCN {TARGET['ins_pcn']}; Group {TARGET['ins_group']}.

PRESCRIBER: {TARGET['rx_prescriber']}; NPI {TARGET['rx_npi']};
  DEA {TARGET['rx_dea']}; Phone {TARGET['rx_phone']}.

PRESCRIPTION: {TARGET['drug_name']} {TARGET['drug_strength']} {TARGET['drug_form']};
  NDC {TARGET['drug_ndc']}; Route Oral; Schedule {TARGET['drug_sched']}
  (controlled -- the DEA above is required); Date written {TARGET['drug_datewritten']};
  Sig "{TARGET['drug_sig']}"; Quantity {TARGET['drug_qty']}; Days supply {TARGET['drug_days']};
  Refills {TARGET['drug_refills']}; DAW code 0.

DISPENSE & REVIEW: Dispense date {TARGET['disp_date']}; NDC dispensed {TARGET['disp_ndc']};
  Qty dispensed {TARGET['disp_qty']}; RPh initials {TARGET['disp_rph']};
  Counseling offered = {TARGET['counsel']}; Interaction check completed = {TARGET['ddi_done']};
  tick the attestation checkbox; then click "Fill prescription".

When the confirmation screen shows an Rx number, you are done."""


def _norm(v):
    return " ".join(str(v).strip().lower().split())


def score(state: dict) -> dict:
    """Compare final form state to TARGET. Returns a report dict."""
    fields = {}
    for k, want in TARGET.items():
        got = state.get(k)
        if isinstance(want, bool):
            ok = bool(got) == want
        else:
            ok = _norm(got) == _norm(want)
        fields[k] = {"want": want, "got": got, "ok": ok}
    allergens = {_norm(a.get("allergen")) for a in state.get("allergies", [])}
    fields["allergies"] = {
        "want": sorted(TARGET_ALLERGENS),
        "got": sorted(allergens),
        "ok": TARGET_ALLERGENS.issubset(allergens),
    }
    total = len(fields)
    passed = sum(1 for f in fields.values() if f["ok"])
    crit_total = len(CRITICAL)
    crit_passed = sum(1 for k in CRITICAL if fields.get(k, {}).get("ok"))
    return {
        "submitted": bool(state.get("submitted")),
        "rxNumber": state.get("rxNumber"),
        "fields": fields,
        "passed": passed, "total": total,
        "crit_passed": crit_passed, "crit_total": crit_total,
    }


def run_arm(name: str, url: str, model: str, max_iters: int) -> dict:
    use_images = name in ("unpinned", "pinned")
    pin = name == "pinned"
    image_paths = [str(REFS / g) for g in GUIDES] if use_images else None

    browser = BrowserTool()
    tools = ToolCollection(browser, BrowserBatchTool(browser))
    traj = Trajectory(model=model, task=f"[{name}] pharmacy form")
    content = build_task_content(build_task(url), image_paths)

    print(f"\n{'='*70}\nARM: {name}  (images={use_images}, pinned={pin})\n"
          f"trajectory: {traj.dir}\n{'='*70}")
    try:
        sampling_loop(
            model=model, task=content, tools=tools, trajectory=traj,
            max_iters=max_iters, interactive=False, pin_reference_images=pin,
        )
        state = browser._ensure_page().evaluate("() => window.getFormState()")
    finally:
        tools.close()

    rep = score(state)
    rep["arm"] = name
    rep["trajectory"] = str(traj.dir)
    return rep


def serve(directory: Path, port: int) -> tuple[str, socketserver.TCPServer]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/index.html", httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline", "unpinned", "pinned"],
                    choices=["baseline", "unpinned", "pinned"])
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max-iters", type=int, default=120)
    ap.add_argument("--port", type=int, default=8731)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set.")
    for g in GUIDES:
        if not (REFS / g).is_file():
            sys.exit(f"missing guide image: {REFS / g}")

    url, httpd = serve(FIXTURE_DIR, args.port)
    print(f"serving fixture at {url}")
    reports = []
    try:
        for arm in args.arms:
            reports.append(run_arm(arm, url, args.model, args.max_iters))
    finally:
        httpd.shutdown()

    print(f"\n\n{'#'*70}\nSCORECARD  (model={args.model})\n{'#'*70}")
    print(f"{'arm':<10}{'submitted':<11}{'overall':<12}{'critical':<12}{'rx#':<12}")
    for r in reports:
        overall = f"{r['passed']}/{r['total']}"
        crit = f"{r['crit_passed']}/{r['crit_total']}"
        print(f"{r['arm']:<10}{str(r['submitted']):<11}{overall:<12}{crit:<12}{str(r['rxNumber']):<12}")
    # per-arm critical-field misses
    for r in reports:
        misses = [k for k in CRITICAL if not r["fields"].get(k, {}).get("ok")]
        if misses:
            print(f"\n[{r['arm']}] critical misses: {misses}")
            for k in misses:
                f = r["fields"][k]
                print(f"    {k}: want={f['want']!r} got={f['got']!r}")


if __name__ == "__main__":
    main()
