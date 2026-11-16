"""Lower bounds for the offline 3D bin packing problem.

The exported bound is ``strong_lower_bound``: the maximum of a family of valid
lower bounds. Every bound here remains valid when orthogonal rotations are
allowed, which is the general case of this solver.

Bounds implemented:

1. ``volume_lower_bound`` (L0): continuous volume relaxation
   (Martello, Pisinger & Vigo, Operations Research, 2000).
2. ``mt_l2_volume_bound``: the Martello & Toth (1990) L2 bound applied to the
   1D relaxation over item volumes. Any feasible 3D packing induces a feasible
   1D packing of volumes into bins of capacity ``bin.volume``, so this bound
   dominates L0 and stays valid under rotation.
3. ``conflict_clique_bound``: rotation-aware pairwise-incompatibility bound.
   Two axis-aligned boxes can share a bin iff there exists a rotation pair and
   an axis along which their summed dimension fits the bin. Items in a clique
   of the conflict graph must occupy pairwise-distinct bins (generalisation of
   the "large items" bound L1 of Martello, Pisinger & Vigo, 2000, to free
   orthogonal rotation; see also Boschetti, 2004).
4. ``clique_volume_bound``: combines (3) with the residual volume argument:
   clique items open ``|Q|`` bins; the remaining items can, at best, fill the
   leftover space of those bins before opening new ones.
"""

from __future__ import annotations

import math

from adaptive_3dbp.domain.models import BinSpec, ItemSpec, ProblemInstance
from adaptive_3dbp.domain.rotations import fitting_rotation_ids, rotation_dimensions

_EPS = 1e-9


def _safe_ceil(value: float) -> int:
    return math.ceil(value - _EPS)


def volume_lower_bound(instance: ProblemInstance) -> int:
    return max(1, _safe_ceil(instance.total_item_volume / instance.bin.volume))


def mt_l2_volume_bound(instance: ProblemInstance) -> int:
    """Martello-Toth L2 bound on the 1D volume relaxation (valid under rotation)."""
    capacity = instance.bin.volume
    weights = [item.volume for item in instance.items]
    best = max(1, _safe_ceil(sum(weights) / capacity))
    half = capacity / 2.0
    alphas = sorted({0.0, *[w for w in weights if w <= half + _EPS]})
    for alpha in alphas:
        n1_count = 0
        n2_sum = 0.0
        n2_count = 0
        n3_sum = 0.0
        for weight in weights:
            if weight > capacity - alpha + _EPS:
                n1_count += 1
            elif weight > half + _EPS:
                n2_count += 1
                n2_sum += weight
            elif weight >= alpha - _EPS:
                n3_sum += weight
        free_in_n2 = n2_count * capacity - n2_sum
        extra = max(0, _safe_ceil((n3_sum - free_in_n2) / capacity))
        best = max(best, n1_count + n2_count + extra)
    return best


def _items_conflict(item_a: ItemSpec, item_b: ItemSpec, bin_spec: BinSpec) -> bool:
    """True when no rotation pair allows the two items to coexist in one bin.

    For exactly two axis-aligned boxes that individually fit the bin, sharing a
    bin is possible iff some axis admits side-by-side placement, i.e. the sum
    of their dimensions on that axis fits the bin.
    """
    rotations_a = fitting_rotation_ids(item_a, bin_spec)
    rotations_b = fitting_rotation_ids(item_b, bin_spec)
    if not rotations_a or not rotations_b:
        return False
    bin_dims = (bin_spec.length, bin_spec.width, bin_spec.height)
    for rotation_a in rotations_a:
        dims_a = rotation_dimensions(item_a, rotation_a)
        for rotation_b in rotations_b:
            dims_b = rotation_dimensions(item_b, rotation_b)
            for axis in range(3):
                if dims_a[axis] + dims_b[axis] <= bin_dims[axis] + _EPS:
                    return False
    return True


def _greedy_max_clique(adjacency: list[set[int]]) -> list[int]:
    n = len(adjacency)
    best_clique: list[int] = []
    orders = [
        sorted(range(n), key=lambda v: len(adjacency[v]), reverse=True),
        sorted(range(n), key=lambda v: len(adjacency[v])),
        list(range(n)),
    ]
    for order in orders:
        clique: list[int] = []
        for vertex in order:
            if all(other in adjacency[vertex] for other in clique):
                clique.append(vertex)
        if len(clique) > len(best_clique):
            best_clique = clique
    return best_clique


def _conflict_clique(instance: ProblemInstance) -> list[int]:
    items = instance.items
    n = len(items)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _items_conflict(items[i], items[j], instance.bin):
                adjacency[i].add(j)
                adjacency[j].add(i)
    return _greedy_max_clique(adjacency)


def conflict_clique_bound(instance: ProblemInstance) -> int:
    return max(1, len(_conflict_clique(instance)))


def clique_volume_bound(instance: ProblemInstance) -> int:
    """|Q| bins for the clique plus residual-volume bins for the rest.

    Valid, but algebraically it reduces to max(|Q|, L0): the extra term is
    max(0, ceil(total_volume / V) - |Q|). So it adds nothing beyond the
    clique size and the volume bound; it is kept because it states the
    combination explicitly. A stronger variant would pack the residual items
    into the clique bins' free space by dimension, not by volume.
    """
    clique = _conflict_clique(instance)
    if not clique:
        return volume_lower_bound(instance)
    bin_volume = instance.bin.volume
    clique_volume = sum(instance.items[index].volume for index in clique)
    rest_volume = instance.total_item_volume - clique_volume
    free_in_clique_bins = len(clique) * bin_volume - clique_volume
    extra = max(0, _safe_ceil((rest_volume - free_in_clique_bins) / bin_volume))
    return len(clique) + extra


_CACHE_MAX_ENTRIES = 64
_bound_cache: dict[tuple, int] = {}


def _bound_cache_key(instance: ProblemInstance) -> tuple:
    """Content key: the bound depends only on bin and item geometry/rotations."""
    return (
        (instance.bin.length, instance.bin.width, instance.bin.height),
        tuple(
            sorted(
                (item.length, item.width, item.height, str(item.allowed_rotations))
                for item in instance.items
            )
        ),
    )


def strong_lower_bound(instance: ProblemInstance) -> int:
    key = _bound_cache_key(instance)
    cached = _bound_cache.get(key)
    if cached is not None:
        return cached
    bound = max(
        volume_lower_bound(instance),
        mt_l2_volume_bound(instance),
        clique_volume_bound(instance),
    )
    if len(_bound_cache) >= _CACHE_MAX_ENTRIES:
        _bound_cache.pop(next(iter(_bound_cache)))
    _bound_cache[key] = bound
    return bound


# Identity fast path: the decoder scores every candidate, and building the
# content key above is O(n log n). Instances are frozen, so the same object
# always has the same bound; the stored reference guards against id() reuse.
_identity_cache: dict[int, tuple[ProblemInstance, int]] = {}


def simple_lower_bound(instance: ProblemInstance) -> int:
    """Kept for backwards compatibility; now delegates to the strong bound."""
    cached = _identity_cache.get(id(instance))
    if cached is not None and cached[0] is instance:
        return cached[1]
    bound = strong_lower_bound(instance)
    if len(_identity_cache) >= _CACHE_MAX_ENTRIES:
        _identity_cache.pop(next(iter(_identity_cache)))
    _identity_cache[id(instance)] = (instance, bound)
    return bound
