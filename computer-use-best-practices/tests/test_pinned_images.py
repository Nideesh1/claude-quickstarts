"""Pinned reference images survive pruning; live screenshots stay bounded.

These are the cache/pruning invariants behind the --image feature: image blocks
in the opening task message are "pinned" (kept for the whole run) while the
agent's own screenshots are still bounded by the normal sliding window.
"""

from computer_use.formatters import StripImagesAtIntervals, StripOldestImages


def _img(tag):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": tag}}


def _tags(messages):
    out = []
    for msg in messages:
        content = msg["content"]
        if not isinstance(content, list):
            continue
        for b in content:
            if b.get("type") == "image":
                out.append(b["source"]["data"])
            elif b.get("type") == "tool_result":
                for s in b.get("content", []):
                    if isinstance(s, dict) and s.get("type") == "image":
                        out.append(s["source"]["data"])
    return out


def _run(strategy, turns=12):
    """Opening message with 2 pinned refs, then `turns` live screenshots."""
    ref1, ref2 = _img("REF1"), _img("REF2")
    opening = {
        "role": "user",
        "content": [{"type": "text", "text": "workflow"}, ref1, ref2],
    }
    pinned = {id(ref1), id(ref2)}
    messages = [opening]
    strategy.pinned_ids = pinned
    for t in range(turns):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"t{t}"}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{t}", "content": [_img(f"SHOT{t}")]}
                ],
            }
        )
        strategy(messages)
    return _tags(messages)


def test_interval_keeps_pinned_and_bounds_live():
    # min=2, interval=3 -> live kept cycles 2,3,4,2,... ; refs always survive.
    surviving = _run(StripImagesAtIntervals(min_images=2, interval=3), turns=12)
    assert "REF1" in surviving and "REF2" in surviving
    live = [t for t in surviving if t.startswith("SHOT")]
    assert len(live) <= 4, live  # bounded by min + interval - 1
    # the most recent live screenshot is always kept
    assert "SHOT11" in surviving


def test_oldest_keeps_pinned_and_bounds_live():
    surviving = _run(StripOldestImages(keep=3), turns=10)
    assert "REF1" in surviving and "REF2" in surviving
    live = [t for t in surviving if t.startswith("SHOT")]
    assert len(live) == 3, live
    assert live == ["SHOT7", "SHOT8", "SHOT9"]


def test_pins_do_not_count_against_live_budget():
    # Two extra pinned refs must not shrink how many live shots are kept.
    no_pin = _run(StripOldestImages(keep=3), turns=10)
    assert len([t for t in no_pin if t.startswith("SHOT")]) == 3


def test_without_pins_refs_are_pruned():
    # Sanity: same setup but no pinning -> the refs DO get stripped.
    ref1, ref2 = _img("REF1"), _img("REF2")
    messages = [{"role": "user", "content": [{"type": "text", "text": "w"}, ref1, ref2]}]
    strat = StripOldestImages(keep=3)  # pinned_ids defaults to None
    for t in range(10):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{t}", "content": [_img(f"SHOT{t}")]}
                ],
            }
        )
        strat(messages)
    surviving = _tags(messages)
    assert "REF1" not in surviving and "REF2" not in surviving
