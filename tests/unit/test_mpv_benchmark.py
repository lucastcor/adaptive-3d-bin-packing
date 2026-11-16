from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptive_3dbp.benchmarks.mpv import (
    Lrand48,
    MPV_CLASSES,
    MPV_SIZES,
    generate_mpv_dataset,
    mpv_instance,
    mpv_items,
    mpv_unit_key,
)


def test_lrand48_matches_libc_sequence() -> None:
    # Values checked against macOS libc lrand48 (docs/benchmark_mpv.md).
    rng = Lrand48(1)
    assert [rng.next() for _ in range(3)] == [89400484, 976015093, 1792756325]
    rng = Lrand48(51)
    assert [rng.next() for _ in range(5)] == [1249289545, 1468912438, 1625849874, 1314316226, 2093333220]


@pytest.mark.parametrize(
    ("instance_class", "n", "first_items", "total_volume"),
    [
        (1, 50, [(39, 85, 72), (11, 84, 96), (36, 10, 34)], 10016449),
        (1, 200, [(66, 77, 42), (52, 83, 69), (94, 67, 100)], 34067306),
        (4, 50, [(84, 65, 70), (11, 84, 96), (36, 10, 34)], 14353284),
        (5, 50, [(39, 25, 27), (11, 84, 96), (36, 10, 34)], 6899119),
        (6, 50, [(6, 9, 5), (7, 1, 1), (4, 1, 5)], 8446),
        (7, 50, [(1, 29, 20), (7, 21, 31), (19, 31, 15)], 385896),
        (8, 50, [(46, 39, 75), (27, 21, 61), (94, 81, 35)], 6745876),
        (8, 200, [(33, 26, 22), (42, 44, 52), (69, 92, 74)], 24796658),
    ],
)
def test_regression_table(instance_class: int, n: int, first_items: list, total_volume: int) -> None:
    _side, items = mpv_items(instance_class, n, 1)
    assert items[:3] == first_items
    assert sum(w * h * d for w, h, d in items) == total_volume


def test_all_320_published_instances_hash() -> None:
    # SHA-256 of the 320 instances from OR-Datasets (Martello2000), serialised
    # one line per item; any change to the generator breaks comparability with
    # every published table.
    digest = hashlib.sha256()
    for instance_class in MPV_CLASSES:
        for n in MPV_SIZES:
            for index in range(1, 11):
                _side, items = mpv_items(instance_class, n, index)
                for w, h, d in items:
                    digest.update(f"{instance_class} {n} {index} {w} {h} {d}\n".encode())
    assert digest.hexdigest() == "20ce376f57b0a7906a2bcbd4b3fc6f351f28be4f06361554fe9b80628c390c2d"


def test_bin_sizes_and_fixed_orientation() -> None:
    for instance_class, side in ((1, 100), (6, 10), (7, 40), (8, 100)):
        instance = mpv_instance(instance_class, 50, 1)
        assert (instance.bin.length, instance.bin.width, instance.bin.height) == (side, side, side)
        assert all(item.allowed_rotations == "fixed" for item in instance.items)
        assert all(max(item.dimensions) <= side for item in instance.items)


def test_dataset_tests_on_the_canonical_instances(tmp_path: Path) -> None:
    manifest = generate_mpv_dataset(tmp_path, classes=(1, 4), sizes=(50,))
    indices = {split: {entry["index"] for entry in entries} for split, entries in manifest["splits"].items()}
    assert indices == {"train": set(range(11, 21)), "validation": set(range(21, 26)), "test": set(range(1, 11))}
    assert len(manifest["splits"]["test"]) == 2 * 10
    stored = json.loads((tmp_path / "manifest.json").read_text())
    first = stored["splits"]["train"][0]
    assert Path(first["path"]).exists()
    with pytest.raises(ValueError):
        generate_mpv_dataset(tmp_path / "bad", classes=(1,), sizes=(50,), splits={"train": (1, 2), "test": (2,)})


def test_dataset_rejects_seed_overlap_between_splits(tmp_path: Path) -> None:
    # seed = v + n: (n=50, v=51) has the seed of the canonical (n=100, v=1),
    # so its 50 items are the first 50 items of that test instance.
    with pytest.raises(ValueError, match="share generator seeds"):
        generate_mpv_dataset(
            tmp_path, classes=(1,), sizes=(50, 100), splits={"train": (51,), "test": (1,)}
        )
    _side, train_items = mpv_items(1, 50, 51)
    _side, test_items = mpv_items(1, 100, 1)
    assert train_items == test_items[:50]


def test_unit_key_groups_classes_sharing_a_seed() -> None:
    assert mpv_unit_key("mpv-c1-n50-v03") == mpv_unit_key("mpv-c8-n50-v03") == "mpv-n50-v03"
    assert mpv_unit_key("mpv-c1-n100-v03") != mpv_unit_key("mpv-c1-n50-v03")
    assert mpv_unit_key("custom-instance") == "custom-instance"
