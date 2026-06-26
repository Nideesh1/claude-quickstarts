# Handoff — pinned reference images for computer use

Context for picking up this work in a fresh session (e.g. on another machine).

## Goal

Let a computer-use run carry a **pinned "start set" of reference images** that
stay in context for the whole run — so when the model reaches a complex step
late in a long task, the how-to guidance is still there. The agent's own live
screenshots keep their normal sliding window; the pinned references are exempt
from pruning.

## What shipped (branch `cua-ollama-compat-and-pinned-reference-images`)

Commit `cfa6355` — the feature:
- `constants.py` — `claude-opus-4-8` registered (enum, `high` default effort,
  effort + autocompaction support sets).
- `__main__.py` — `--image PATH` flag (repeatable). Builds the opening user
  message as `[task text, "Reference image N (name):", <image block>, …]`.
  Images are raw base64 (stdlib, no resize — they aren't click targets).
- `loop.py` — image blocks in the opening message are "pinned" by object id and
  passed to the pruner.
- `formatters.py` — `_image_slots` + both strip classes skip pinned ids
  (including the MB force-prune path). Pinned images don't count against the
  live sliding-window budget.
- `render.py` — `info()` helper.
- Also bundled: Ollama compatibility (`relay_images_top_level`, `_split_out_media`,
  `_as_xy` string-coordinate coercion) + `.env.ollama.example`.

Commit `a0cb92c` — the test work:
- `loop.py` — `pin_reference_images: bool = True` flag so the harness can A/B
  pinning on vs off with identical data.
- `tests/test_pinned_images.py` — fast unit test (no API/browser): pinned refs
  survive pruning, live screenshots stay bounded, pins don't shrink the live
  budget. Passing.
- `tests/fixtures/pharmacy/index.html` — realistic 6-step pharmacy dispensing
  form (~50 fields, format validation, dynamic allergy rows). Test hooks:
  `window.getFormState()` (full record + deterministic Rx#), `window.gotoStep(n)`.
- `tests/fixtures/pharmacy/refs/guide-*.png` — 3 annotated how-to screenshots
  (allergies / prescription / review) — the pinned "start set".
- `tests/pharmacy_live/run_harness.py` + `README.md` — live 3-arm harness.

## The test idea

Three arms, same task/data, scored against the target form state:

| arm        | refs | pinning | expected |
|------------|------|---------|----------|
| `baseline` | none | —       | fumbles tricky steps |
| `unpinned` | yes  | OFF     | starts ok, **fails late** (guides pruned before step 6) |
| `pinned`   | yes  | ON      | nails the late step |

Harness sets a small `CU_IMAGE_PRUNE_INTERVAL` so unpinned guides are stripped
before the late complex step (step 6: required radios + attestation). If only
`pinned` scores well, pinning (not just "having images") is what helped.

## How to run

```bash
cd computer-use-best-practices
pip install -r requirements.txt
playwright install chromium                 # agent drives headless Chromium
cp .env.example .env                         # add a real sk-ant-... key
python tests/pharmacy_live/run_harness.py    # all 3 arms, opus-4-8
python -m pytest tests/test_pinned_images.py # fast unit test, no key needed
```

## Status / what's left

- ✅ Feature + Ollama compat committed and pushed.
- ✅ Fixture, guide images, unit test (passing), harness — committed and pushed.
- ⬜ **The live 3-arm run has not produced a scorecard yet.** It was blocked on
  the dev machine by **Google Santa** (binary allowlisting) killing headless
  Chromium at launch — `TargetClosedError: ... browser has been closed` right
  after `<launched> pid=...`. Not a code bug. Run on an unmanaged machine.
- ⬜ After a successful run, sanity-check the scorecard: confirm `pinned` beats
  `unpinned` on the critical late-step fields (`counsel`, `ddi_done`, `attest`,
  `drug_sched`, `rx_dea`, dropdowns). If `unpinned` already passes, lower
  `CU_IMAGE_PRUNE_INTERVAL` / lengthen the task so the guides truly get pruned
  before they're needed.

## Notes

- The repo defaults to first-party Anthropic. The Ollama path is opt-in via env
  (`ANTHROPIC_BASE_URL` + `CU_RELAY_IMAGES_TOP_LEVEL=true`); see `.env.ollama.example`.
- `runs/` (trajectories) is gitignored; each harness arm writes one for replay
  in the trajectory viewer (`dev_ui/trajectory_viewer`).
