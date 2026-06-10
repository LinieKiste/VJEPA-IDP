"""EgoPER access + frame-sampling helpers for video-QA inference baselines.

EgoPER lives at ``datasets/egoper/`` with one sub-dir per task (Coffee, Tea, ...),
each holding ``trim_videos/*.mp4`` (the full ~10-12 min recordings) plus frame zips.
Per-video ground truth is in the top-level ``annotation.json``:

    annotation[task]["segments"] = [
        {"video_id": "coffee_u1_a1_error_001",
         "labels": {"action": [int, ...],          # action id per segment
                    "action_type": [int, ...],     # 0=Normal,1=Modification,2=Slip,3=Correction,4=Addition
                    "time_stamp": [[start, end], ...],
                    "error_description": [str, ...]}},  # action name for normal segs, error text for errors
        ...]

So a video has an error iff any ``action_type != 0``; ``error_description`` +
``time_stamp`` give what/when, which lets us score free-text QA answers.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

EGOPER_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "egoper"

# task key (as in annotation.json) -> on-disk sub-directory name
TASK_DIRS = {
    "coffee": "Coffee",
    "pinwheels": "Pinwheels",
    "oatmeal": "Oatmeal",
    "quesadilla": "Quesadilla",
    "tea": "Tea",
}


@functools.lru_cache(maxsize=1)
def _annotation() -> dict:
    return json.loads((EGOPER_ROOT / "annotation.json").read_text())


def tasks() -> list[str]:
    return list(TASK_DIRS)


@functools.lru_cache(maxsize=None)
def _idx2type(task: str) -> dict[int, str]:
    return {v: k for k, v in _annotation()[task]["actiontype2idx"].items()}


def video_path(task: str, video_id: str) -> Path:
    return EGOPER_ROOT / TASK_DIRS[task] / "trim_videos" / f"{video_id}.mp4"


def _segment(task: str, video_id: str) -> dict:
    for s in _annotation()[task]["segments"]:
        if s["video_id"] == video_id:
            return s
    raise KeyError(f"{video_id} not in annotation for task {task!r}")


def list_videos(task: str, kind: str | None = None, existing_only: bool = True) -> list[str]:
    """Video ids for a task. ``kind`` in {None, 'error', 'normal'}.

    ``existing_only`` keeps only ids whose .mp4 is present in trim_videos (many
    segments in annotation.json only ship as frame zips, not trimmed videos).
    """
    out = []
    for s in _annotation()[task]["segments"]:
        vid = s["video_id"]
        if kind and kind not in vid:
            continue
        if existing_only and not video_path(task, vid).exists():
            continue
        out.append(vid)
    return out


def ground_truth(task: str, video_id: str) -> dict:
    """Structured GT for a video: error flag, error segments, and full timeline."""
    labels = _segment(task, video_id)["labels"]
    i2t = _idx2type(task)
    timeline, errors = [], []
    for act, atype, ts, desc in zip(
        labels["action"], labels["action_type"], labels["time_stamp"], labels["error_description"]
    ):
        row = {"start": ts[0], "end": ts[1], "type": i2t.get(atype, str(atype)), "desc": desc}
        timeline.append(row)
        if atype != 0:
            errors.append(row)
    return {
        "video_id": video_id,
        "task": task,
        "has_error": bool(errors),
        "errors": errors,
        "timeline": timeline,
    }


def describe_gt(task: str, video_id: str) -> str:
    """Human-readable GT summary (for printing next to model answers)."""
    gt = ground_truth(task, video_id)
    head = f"{video_id}  |  has_error={gt['has_error']}"
    if not gt["has_error"]:
        return head + "  (normal recording)"
    lines = [head, "  errors:"]
    for e in gt["errors"]:
        lines.append(f"    [{e['start']:6.1f}-{e['end']:6.1f}s] {e['type']}: {e['desc']}")
    return "\n".join(lines)


def normal_actions(task: str) -> dict[int, str]:
    """Map action id -> readable step description from ``<task>_normal_actions.txt``.

    The ``Action_N`` index equals the action id in ``annotation.json``'s ``action2idx``.
    """
    path = EGOPER_ROOT / TASK_DIRS[task] / f"{task}_normal_actions.txt"
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, desc = line.split(" ", 1)  # "Action_5", "Put_dripper_cone_on_mug"
        out[int(key.split("_")[1])] = desc.replace("_", " ")
    return out


@functools.lru_cache(maxsize=None)
def task_graph(task: str) -> dict:
    """Parse the procedure DAG for a task from the top-level ``task_graph.txt``.

    Returns ``{"edges": [(a, b), ...], "start": int, "end": int}`` where node
    indices are action ids. The graph has parallel branches that converge.
    """
    import ast
    import re

    text = (EGOPER_ROOT / "task_graph.txt").read_text()
    name = re.escape(TASK_DIRS[task])
    m = re.search(
        rf"^{name}:\s*\nEdges:\s*(\[.*?\])\s*\nStart node:\s*(\d+)\s*\nEnd node:\s*(\d+)",
        text,
        re.M,
    )
    if not m:
        raise KeyError(f"no task graph block for {task!r}")
    return {"edges": ast.literal_eval(m.group(1)), "start": int(m.group(2)), "end": int(m.group(3))}


def _toposort(edges, start, end) -> list[int]:
    """Deterministic topological order (Kahn's algorithm) over the DAG nodes."""
    from collections import defaultdict, deque

    nodes = {start, end}
    adj = defaultdict(list)
    indeg = defaultdict(int)
    for a, b in edges:
        nodes |= {a, b}
        adj[a].append(b)
        indeg[b] += 1
    for n in nodes:
        indeg.setdefault(n, 0)
    q = deque(sorted(n for n in nodes if indeg[n] == 0))
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    return order


# Short, paper-accurate explanation of the task graph + error taxonomy (EgoPER,
# Lee et al. CVPR 2024, Sec. 3.1 & error taxonomy). Prepended to the step list so the
# model knows what the procedure is and what counts as an error.
_TASK_GRAPH_NOTE = (
    "This task is defined by a task graph: a set of steps with ordering constraints that "
    "encodes every valid way to complete the recipe. A correct execution performs each "
    "step according to its description while respecting the required order; steps with no "
    "ordering constraint between them may be done in any order. An error is any deviation "
    "from the task graph, of these kinds: Omission (a required step is skipped), Addition "
    "(an unnecessary extra step not in the recipe), Modification (a step done with the "
    "wrong tool or ingredient), or Slip (a step executed incorrectly so its goal is not "
    "achieved, e.g. spilling or using the wrong container)."
)


def procedure_text(task: str) -> str:
    """Readable 'intended procedure' for a task, to ground the model's error checking.

    Prepends a paper-accurate explanation of the task graph + error types, then lists the
    steps (graph nodes named from ``normal_actions``) in one valid topological order.
    The graph encodes *all* valid orderings, so the numbering is only one such order —
    only the precedence constraints are binding. Nodes without a description (BG /
    terminal) are dropped.
    """
    acts = normal_actions(task)
    g = task_graph(task)
    steps = [acts[n] for n in _toposort(g["edges"], g["start"], g["end"]) if n in acts]
    lines = [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    return (
        _TASK_GRAPH_NOTE
        + f"\n\nSteps of the correct procedure for {task} (shown in one valid order; only "
        "the graph's precedence constraints are required, not this exact numbering):\n"
        + "\n".join(lines)
    )


def sample_frames(path, num_frames: int = 32):
    """Uniformly sample ``num_frames`` frames from a video.

    Returns ``(frames, timestamps)`` where frames is a uint8 array (T, H, W, 3)
    and timestamps is a list of seconds. NOTE: uniform sampling over a ~12-min
    EgoPER clip means ~1 frame / 23 s at 32 frames — brief/subtle errors may fall
    between samples. Raise num_frames (VRAM permitting) or sample around a window
    of interest for finer temporal coverage.
    """
    import numpy as np
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0))
    n = len(vr)
    fps = float(vr.get_avg_fps()) or 30.0
    idx = np.linspace(0, n - 1, num=min(num_frames, n)).astype(int)
    frames = vr.get_batch(idx).asnumpy()
    timestamps = [round(int(i) / fps, 2) for i in idx]
    return frames, timestamps
