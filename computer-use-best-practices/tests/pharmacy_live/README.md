# Pinned reference-images live harness

Proves that pinning reference images (the `--image` feature) actually helps a
real agent run, using a local fake pharmacy form.

## What it does

Runs the agent against `tests/fixtures/pharmacy/index.html` (a 6-step dispensing
record) in up to three arms and scores the final form state vs the target data:

| arm        | reference images | pinning | expectation |
|------------|------------------|---------|-------------|
| `baseline` | none             | —       | fumbles the tricky steps |
| `unpinned` | 3 guides         | OFF     | starts well, **fails late** (guides pruned before step 6) |
| `pinned`   | 3 guides         | ON      | nails the late step (guides kept all run) |

The guides (`tests/fixtures/pharmacy/refs/guide-*.png`) are annotated how-to
screenshots for the non-obvious controls (allergy rows, dropdowns, and the
required radios/attestation on the final Review step). The harness sets a small
image-prune interval so the `unpinned` guides are stripped before step 6 — which
is exactly where pinning should pay off. If only `pinned` scores well, the
feature is proven.

## Run

```bash
# from the computer-use-best-practices dir, with a real ANTHROPIC_API_KEY in .env
pip install -r requirements.txt
playwright install chromium          # required — the agent drives headless Chromium

python tests/pharmacy_live/run_harness.py                  # all 3 arms, opus-4-8
python tests/pharmacy_live/run_harness.py --arms pinned     # one arm
python tests/pharmacy_live/run_harness.py --model claude-sonnet-4-6
```

Prints a scorecard (submitted / overall fields / critical fields / Rx#) plus the
critical-field misses per arm.

## Note

Must run on a machine where headless Chromium is allowed to execute. On
binary-allowlisted Macs (e.g. **Google Santa**), the Chromium process is killed
the instant it launches — you'll see `TargetClosedError: ... browser has been
closed` right after `<launched> pid=...`. That's the environment blocking the
binary, not a code bug. Run on an unmanaged machine.

## Fast unit test (no API, no browser)

`tests/test_pinned_images.py` checks the pruning invariants directly (pinned
survive, live screenshots stay bounded). Runs in normal `pytest`.
