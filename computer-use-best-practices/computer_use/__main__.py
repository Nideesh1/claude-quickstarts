"""CLI entrypoint: `python -m computer_use "do something"`."""

import argparse
import base64
import mimetypes
from pathlib import Path
from typing import Any, get_args

from constants import (
    ADVISOR_PROMPT_ADDENDUM,
    SCRATCH_PROMPT_ADDENDUM,
    SYSTEM_PROMPT,
    Model,
    ThinkingEffort,
    cfg,
)

from . import render
from .loop import sampling_loop
from .preflight import check_and_warn
from .tools import ToolCollection
from .tools.base import Tool
from .tools.batch import BrowserBatchTool, ComputerBatchTool
from .tools.browser import BrowserTool
from .tools.computer import ComputerTool
from .tools.editor import EditorTool
from .tools.open_app import OpenApplicationTool
from .tools.shell import BashTool, PythonTool
from .trajectory import Trajectory


def build_tools(scratch_dir: Path | None = None) -> ToolCollection:
    if not (cfg.enable_computer_use_tools or cfg.enable_browser_use_tools):
        raise ValueError(
            "At least one of cfg.enable_computer_use_tools or "
            "cfg.enable_browser_use_tools must be True."
        )
    tools: list[Tool] = []
    if cfg.enable_computer_use_tools:
        computer = ComputerTool()
        tools += [computer, ComputerBatchTool(computer), OpenApplicationTool()]
    if cfg.enable_browser_use_tools:
        browser = BrowserTool()
        tools += [browser, BrowserBatchTool(browser)]
    tools += [BashTool(scratch_dir), PythonTool(scratch_dir)]
    if cfg.enable_editor_tool and scratch_dir is not None:
        tools.append(EditorTool(scratch_dir))
    return ToolCollection(*tools)


# Image media types the Anthropic Messages API accepts as image blocks.
_SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
# File extensions globbed by --image-dir (each file is validated by media type in build_task_content).
_SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def collect_pinned_images(image_paths: list[str] | None, image_dir: str | None) -> list[str]:
    """Merge explicit ``--image`` paths with every image found in ``--image-dir``.

    The directory is the "drop your pinned references here" folder: any image in
    it is pinned for the whole run, exactly like passing it with ``--image``.
    """
    paths: list[str] = list(image_paths or [])
    if image_dir:
        d = Path(image_dir)
        if not d.is_dir():
            raise SystemExit(f"--image-dir: not a directory: {image_dir}")
        found = sorted(
            p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTS
        )
        if not found:
            raise SystemExit(f"--image-dir {image_dir}: no images found (.png/.jpg/.gif/.webp)")
        print(f"pinning {len(found)} image(s) from {image_dir}/: {', '.join(p.name for p in found)}")
        paths += [str(p) for p in found]
    return paths


def build_task_content(task: str, image_paths: list[str] | None) -> str | list[dict[str, Any]]:
    """Build the opening user message.

    With no images this is just the task string. With ``--image`` paths it
    becomes a content-block list: the task text, then each reference image
    preceded by a short label. Images are sent as raw bytes (no resize/re-encode
    -- they're for the model to look at, not to map click coordinates back to),
    and any image in this opening message is pinned by the loop so the pruner
    never strips it.
    """
    if not image_paths:
        return task
    blocks: list[dict[str, Any]] = [{"type": "text", "text": task}]
    for n, raw in enumerate(image_paths, 1):
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"--image: file not found: {raw}")
        media_type, _ = mimetypes.guess_type(path.name)
        if media_type not in _SUPPORTED_IMAGE_TYPES:
            raise SystemExit(
                f"--image {raw}: unsupported type {media_type!r}; "
                f"use one of {sorted(_SUPPORTED_IMAGE_TYPES)}"
            )
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        blocks.append({"type": "text", "text": f"Reference image {n} ({path.name}):"})
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        )
    return blocks


def build_system_prompt(scratch_dir: Path | None) -> str:
    prompt = SYSTEM_PROMPT
    if cfg.enable_editor_tool and scratch_dir is not None:
        prompt += SCRATCH_PROMPT_ADDENDUM.format(scratch_dir=scratch_dir)
    if cfg.enable_advisor_tool:
        prompt += ADVISOR_PROMPT_ADDENDUM
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(prog="computer_use")
    parser.add_argument("task", help="natural-language task for the agent")
    parser.add_argument(
        "--image",
        action="append",
        metavar="PATH",
        help="reference image to attach to the task (repeatable). These are "
        "pinned in context for the whole run -- never pruned -- to guide the "
        "model's workflow.",
    )
    parser.add_argument(
        "--image-dir",
        metavar="DIR",
        help="folder of reference images; every image in it (.png/.jpg/.gif/.webp) "
        "is pinned for the whole run, same as passing each with --image. Combines "
        "with --image.",
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in Model] + list(cfg.extra_models),
        default=Model.SONNET_4_6.value,
    )
    parser.add_argument("--max-iters", type=int, default=cfg.default_max_iters)
    parser.add_argument(
        "--thinking",
        choices=list(get_args(ThinkingEffort)),
        default=None,
        help="reasoning effort; overrides the per-model default in constants.py",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the macOS Screen Recording / Accessibility permission check",
    )
    args = parser.parse_args()

    render.safety_banner()

    if not args.skip_preflight:
        check_and_warn(require=True)

    image_paths = collect_pinned_images(args.image, args.image_dir)
    task_content = build_task_content(args.task, image_paths or None)

    traj = Trajectory(model=args.model, task=args.task)
    system_prompt = build_system_prompt(traj.scratch_dir)
    (traj.dir / "system_prompt.txt").write_text(system_prompt)
    tools = build_tools(scratch_dir=traj.scratch_dir)
    print(f"trajectory: {traj.dir}")

    try:
        sampling_loop(
            model=args.model,
            task=task_content,
            tools=tools,
            trajectory=traj,
            system_prompt=system_prompt,
            thinking_effort=args.thinking,
            max_iters=args.max_iters,
        )
    finally:
        tools.close()


if __name__ == "__main__":
    main()
