"""Martello, Pisinger & Vigo (2000) 3D-BPP benchmark, generated bit-exactly.

Reimplementation of the instance generator in Pisinger's ``test3dbpp.c``
("new3dbpp" version, the one with ``unsigned int`` state that every public
mirror uses). Validated against the 320 published instances (classes 1-8,
n in {50, 100, 150, 200}, 10 instances each) of the OR-Datasets repository:
320/320 identical, item by item. See docs/benchmark_mpv.md for the
specification, sources and pitfalls. The original C code is distributed
"free of charge for research and academic purposes only"; this module is an
independent Python implementation of the published procedure.

Specification in short:

- RNG: POSIX ``lrand48`` (48-bit LCG, output X >> 17), ``srand48(v + n)`` for
  instance v = 1..10 (the seed does NOT depend on the class).
- ``rint(a, b) = lrand48() % (b - a + 1) + a`` (plain modulo, keep the bias).
- Per item, in order: for classes 1-5 draw ``t = rint(1, 10)`` and use type
  t if t <= 5, else the class itself (P(class type) = 0.6, others 0.1 each);
  then draw w, h, d in that order, with C integer division (2*100/3 = 66).
- Bins: 100^3 for classes 1-5 and 8, 10^3 for class 6, 40^3 for class 7.
- Axes: w -> x (length), h -> y (width), d -> z (height). The benchmark has
  no gravity; "height" here is just the third axis.

Published heuristic results (GLS, TS2PACK, EP, GRASP/VND, BRKGA) use FIXED
orientation, hence ``rotations="fixed"`` by default. Because the seed ignores
the class, about half of the items coincide across classes 1-5 for the same
(n, v): split datasets by instance index v, never by class, and pair
statistics by (n, v).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from adaptive_3dbp.domain.models import BinSpec, ItemSpec, ProblemInstance

MPV_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8)
# Classes most papers report (2 and 3 behave like 1).
REPORTED_CLASSES = (1, 4, 5, 6, 7, 8)
MPV_SIZES = (50, 100, 150, 200)
MPV_INSTANCES_PER_GROUP = 10
BIN_SIDE = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 10, 7: 40, 8: 100}
# Split by instance index v (see module docstring). The 10 canonical
# instances (v = 1..10, the ones in every published table) are the test set;
# training and validation use further instances of the same generator. Seeds
# are v + n, so (n, v) pairs of different splits must never share a seed: an
# instance with seed s and n1 items is a prefix of the one with seed s and
# n2 > n1 items of the same class. generate_mpv_dataset checks this.
DEFAULT_SPLITS = {
    "train": tuple(range(11, 21)),
    "validation": tuple(range(21, 26)),
    "test": tuple(range(1, 11)),
}

RotationMode = Literal["fixed", "all_orthogonal"]


class Lrand48:
    """POSIX lrand48: X_{k+1} = (0x5DEECE66D X_k + 0xB) mod 2^48, output X >> 17."""

    _A = 0x5DEECE66D
    _C = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self.state = ((seed & 0xFFFFFFFF) << 16) | 0x330E

    def next(self) -> int:
        self.state = (self._A * self.state + self._C) & self._MASK
        return self.state >> 17

    def rint(self, low: int, high: int) -> int:
        return self.next() % (high - low + 1) + low


def _draw_item(rng: Lrand48, side: int, instance_class: int) -> tuple[int, int, int]:
    W = H = D = side
    item_type = instance_class
    if instance_class <= 5:
        t = rng.rint(1, 10)
        if t <= 5:
            item_type = t
    # Statement order matters: w, then h, then d (separate C statements).
    if item_type == 1:
        w = rng.rint(1, W // 2)
        h = rng.rint(2 * H // 3, H)
        d = rng.rint(2 * D // 3, D)
    elif item_type == 2:
        w = rng.rint(2 * W // 3, W)
        h = rng.rint(1, H // 2)
        d = rng.rint(2 * D // 3, D)
    elif item_type == 3:
        w = rng.rint(2 * W // 3, W)
        h = rng.rint(2 * H // 3, H)
        d = rng.rint(1, D // 2)
    elif item_type == 4:
        w = rng.rint(W // 2, H)  # sic: upper bound H in the original; W == H here
        h = rng.rint(H // 2, H)
        d = rng.rint(D // 2, D)
    elif item_type == 5:
        w = rng.rint(1, W // 2)
        h = rng.rint(1, H // 2)
        d = rng.rint(1, D // 2)
    elif item_type == 6:
        w, h, d = rng.rint(1, 10), rng.rint(1, 10), rng.rint(1, 10)
    elif item_type == 7:
        w, h, d = rng.rint(1, 35), rng.rint(1, 35), rng.rint(1, 35)
    elif item_type == 8:
        w, h, d = rng.rint(1, 100), rng.rint(1, 100), rng.rint(1, 100)
    else:  # pragma: no cover - guarded by mpv_items
        raise ValueError(f"unknown item type {item_type}")
    return w, h, d


def mpv_items(instance_class: int, n: int, index: int) -> tuple[int, list[tuple[int, int, int]]]:
    """Bin side and the n items (w, h, d) of instance ``index`` (1-based) of a class."""
    if instance_class not in BIN_SIDE:
        raise ValueError(f"MPV class must be 1..8, got {instance_class}")
    if n < 1 or index < 1:
        raise ValueError("n and index must be positive")
    side = BIN_SIDE[instance_class]
    rng = Lrand48(index + n)
    return side, [_draw_item(rng, side, instance_class) for _ in range(n)]


def mpv_instance_id(instance_class: int, n: int, index: int) -> str:
    return f"mpv-c{instance_class}-n{n}-v{index:02d}"


_MPV_ID_RE = re.compile(r"^mpv-c(\d+)-n(\d+)-v(\d+)$")


def mpv_unit_key(instance_id: str) -> str:
    """Statistical unit of an MPV instance: the (n, v) pair, across classes.

    The RNG seed is v + n for every class, so instances of classes 1-5 with
    the same (n, v) share about half of their items and class 6 is a function
    of class 8. They are not independent observations; paired tests must
    aggregate them into one unit. Non-MPV ids are their own unit.
    """
    match = _MPV_ID_RE.match(instance_id)
    if not match:
        return instance_id
    _instance_class, n, index = match.groups()
    return f"mpv-n{int(n)}-v{int(index):02d}"


def mpv_instance(
    instance_class: int,
    n: int,
    index: int,
    rotations: RotationMode = "fixed",
) -> ProblemInstance:
    side, items = mpv_items(instance_class, n, index)
    return ProblemInstance(
        instance_id=mpv_instance_id(instance_class, n, index),
        bin=BinSpec(length=side, width=side, height=side),
        items=[
            ItemSpec(id=f"item-{position:03d}", length=w, width=h, height=d, allowed_rotations=rotations)
            for position, (w, h, d) in enumerate(items)
        ],
    )


def generate_mpv_dataset(
    output_dir: Path,
    classes: tuple[int, ...] = REPORTED_CLASSES,
    sizes: tuple[int, ...] = MPV_SIZES,
    splits: dict[str, tuple[int, ...]] | None = None,
    rotations: RotationMode = "fixed",
) -> dict[str, Any]:
    """Write the instances and a campaign-compatible manifest (train/validation/test).

    Splits are by instance index v, identical for every (class, n), so items
    shared across classes 1-5 never cross splits. No solver run is involved:
    the dataset is a pure function of its arguments.
    """
    splits = {name: tuple(indices) for name, indices in (splits or DEFAULT_SPLITS).items()}
    used = [index for indices in splits.values() for index in indices]
    if len(used) != len(set(used)):
        raise ValueError("an instance index cannot belong to two splits")
    seeds_by_split = {name: {index + n for index in indices for n in sizes} for name, indices in splits.items()}
    names = list(seeds_by_split)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            shared = seeds_by_split[left] & seeds_by_split[right]
            if shared:
                raise ValueError(
                    f"splits {left!r} and {right!r} share generator seeds {sorted(shared)[:5]} "
                    "(seed = v + n): their instances would overlap as prefixes"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "benchmark": "martello_pisinger_vigo_2000",
        "generator": "adaptive_3dbp.benchmarks.mpv (bit-exact port of test3dbpp.c, new3dbpp)",
        "rotations": rotations,
        "classes": [f"mpv-c{instance_class}-n{n}" for instance_class in classes for n in sizes],
        "split_by": "instance_index",
        "split_indices": {name: list(indices) for name, indices in splits.items()},
        "statistical_unit": "mpv (n, v)",
        "splits": {name: [] for name in splits},
    }
    for split, indices in splits.items():
        for instance_class in classes:
            for n in sizes:
                for index in indices:
                    instance = mpv_instance(instance_class, n, index, rotations=rotations)
                    path = instances_dir / f"{instance.instance_id}.json"
                    path.write_text(instance.model_dump_json(indent=2), encoding="utf-8")
                    manifest["splits"][split].append(
                        {
                            "path": str(path),
                            "instance_id": instance.instance_id,
                            "class": f"mpv-c{instance_class}-n{n}",
                            "mpv_class": instance_class,
                            "n": n,
                            "index": index,
                        }
                    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
