// Native simulated-annealing core for offline 3D bin packing.
//
// This file mirrors the Python reference implementation and must stay in
// exact parity with it:
//   - decoder:   src/adaptive_3dbp/decoders/extreme_points.py
//   - checks:    src/adaptive_3dbp/domain/validation.py, domain/rotations.py
//   - energy:    src/adaptive_3dbp/domain/scoring.py
//   - pool:      src/adaptive_3dbp/constructors/pool.py
//   - search:    src/adaptive_3dbp/metaheuristics/simulated_annealing/
//                {solver,schedule,operators}.py
//
// The decoder is deterministic and reproduces the Python placements exactly
// (tests/unit/test_native_parity.py). That is why the arithmetic keeps the
// Python evaluation order, emulates CPython's float sum(), calls the platform
// pow() for u**2, rounds dedupe keys half-to-even like round(), and the build
// disables FMA contraction (-ffp-contract=off). The SA follows the same rules
// as the Python solver but draws from its own RNG stream (mt19937_64), so
// search trajectories differ from random.Random.
//
// Portability: standard library only, stdin/stdout only, single thread, so it
// can be compiled to WebAssembly with Emscripten.
//
// ---------------------------------------------------------------------------
// Protocol v2 (whitespace-separated tokens on stdin; "none" = Python None)
//
//   v2
//   mode solve|decode
//   sum_mode neumaier|naive        CPython >= 3.12 sum() is compensated
//   policy min_z_y_x|contact_then_z|min_waste
//   bin <length> <width> <height> <max_weight|none>
//   support <support_ratio_min|none> <stability_required 0|1>
//   items <n>
//     n rows: <l> <w> <h> <weight|none> <k> <rotation ids x k>
//             (fixed -> 1 0, all_orthogonal -> 6 0 1 2 3 4 5, list -> as is)
//   min_distance <m>
//     m rows: <item index a> <item index b> <distance>
//   lower_bound <bins>              (solve) stop once best bins <= this
//   seed <uint64>                   (solve) start k uses seed + k * 9973
//   <SolverConfig field> <value>    (solve) see read_config_field()
//   operators <k> (<name> <probability>) x k     (solve, dict order)
//   initial_pool <k> (<n order ids> <n rotation ids>) x k
//                                   (solve, optional; built here when absent)
//   candidate <k> <order ids x k> <m> <rotation ids x m>    (decode)
//   end
//
// Output: one JSON object on stdout. "status" is "feasible", "infeasible" or
// "error" (protocol error, exit code 1).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double EPS = 1e-9;
// schedule.py: fallback T0 is FALLBACK_TEMPERATURE / bins of the start
// solution; the calibration walk takes at most CALIBRATION_MAX_FACTOR steps
// per wanted usable delta.
constexpr double FALLBACK_TEMPERATURE = 0.05;
constexpr int CALIBRATION_MAX_FACTOR = 4;
constexpr std::uint64_t START_SEED_STRIDE = 9973;

const int ROTATIONS[6][3] = {
    {0, 1, 2},
    {0, 2, 1},
    {1, 0, 2},
    {1, 2, 0},
    {2, 0, 1},
    {2, 1, 0},
};

using Clock = std::chrono::steady_clock;
using Dims = std::array<double, 3>;
using Rank = std::array<double, 5>;
using Score = std::array<double, 3>;

enum class Policy { MinZYX, ContactThenZ, MinWaste };

struct ProtocolError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// CPython's built-in sum() over floats: Neumaier compensated summation since
// 3.12, plain left-to-right addition before. Bin occupancies, the waste term
// of the placement rank and the energy are sums, so the mode must match the
// interpreter that runs the Python reference.
struct PySum {
  bool compensated;
  double hi = 0.0;
  double lo = 0.0;

  explicit PySum(bool compensated_mode) : compensated(compensated_mode) {}

  void add(double x) {
    if (!compensated) {
      hi += x;
      return;
    }
    double t = hi + x;
    if (std::fabs(hi) >= std::fabs(x)) {
      lo += (hi - t) + x;
    } else {
      lo += (x - t) + hi;
    }
    hi = t;
  }

  double value() const {
    if (compensated && lo != 0.0 && std::isfinite(lo)) {
      return hi + lo;
    }
    return hi;
  }
};

// Python's u**2 calls the platform pow(), which is not always the correctly
// rounded u*u. The volatile exponent stops the compiler from folding
// pow(u, 2.0) into a multiplication.
double py_square(double value) {
  volatile double two = 2.0;
  return std::pow(value, two);
}

struct Item {
  Dims dims{};
  double volume = 0.0;
  bool has_weight = false;
  double weight = 0.0;           // item_weight(): None -> 0.0
  std::vector<int> allowed_spec;  // rotation ids as declared
  std::vector<int> fitting;       // fitting_rotation_ids()
};

struct MinDistanceRule {
  int a;
  int b;
  double distance;
};

struct Instance {
  double bin_l = 0.0;
  double bin_w = 0.0;
  double bin_h = 0.0;
  double bin_volume = 0.0;
  bool has_max_weight = false;
  double max_weight = 0.0;
  double min_support = 0.0;  // required_support_ratio()
  std::vector<Item> items;
  std::vector<MinDistanceRule> min_distance;
  std::vector<double> required_distance;  // n x n, first matching rule wins
  Policy policy = Policy::MinZYX;
  bool compensated_sum = true;
};

struct Point {
  double x;
  double y;
  double z;
};

struct Placement {
  int item_index;
  int bin_id;
  double x;
  double y;
  double z;
  int rotation_id;
  double l;
  double w;
  double h;
  double x2;  // max_corner
  double y2;
  double z2;
  double volume;
};

struct Candidate {
  std::vector<int> order;
  std::vector<int> rotations;
};

struct Unplaced {
  int item_index;
  std::string reason;
};

struct Decoded {
  Candidate candidate;
  std::vector<Placement> placements;
  std::vector<Unplaced> unplaced;
  int bins = 0;
  double fragmentation = 1.0;
  double compactness_mean = 0.0;
  double objective = 1.0;  // E = bins + r
  Score score{0.0, 1.0, -0.0};
};

struct ConvergencePoint {
  int evaluation;
  double elapsed_seconds;
  int bins_used;
  double objective_scalar;
  bool accepted;
  double temperature;
  int current_bins;
  double current_objective_scalar;
  int best_bins;
  double best_objective_scalar;
  double delta;
  bool improved;
};

struct Config {
  std::uint64_t seed = 0;
  int max_evaluations = 2500;
  double time_limit_seconds = 10.0;
  int iterations_per_temperature = 80;
  int stagnation_limit = 600;
  int ruin_size = 4;
  std::optional<double> initial_temperature;
  double target_initial_acceptance = 0.82;
  int temperature_calibration_samples = 24;
  double cooling_rate = 0.96;
  double min_temperature = 1e-9;        // absolute floor of the stop temperature
  double min_temperature_ratio = 1e-3;  // stop at max(min_temperature, ratio * T0)
  double reheating_factor = 1.35;
  int multi_start_count = 3;
  int convergence_sample_rate = 1;
  int lower_bound = 1;
  std::vector<std::pair<std::string, double>> operators = {
      {"swap", 0.32}, {"relocate", 0.24}, {"reverse", 0.14}, {"rotate", 0.20}, {"ruin_recreate", 0.10},
  };
};

// ---------------------------------------------------------------------------
// RNG: mt19937_64 is fully specified by the standard, but the std
// distributions are not, so the draws are implemented here to keep runs
// reproducible across standard libraries (libstdc++, libc++, Emscripten).

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed) {}

  // Uniform in [0, 1) with 53 random bits, like random.random().
  double random() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }

  // Uniform integer in [0, n), n >= 1, without modulo bias.
  int below(int n) {
    const std::uint64_t bound = static_cast<std::uint64_t>(n);
    const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t limit = max - max % bound;
    std::uint64_t draw = engine_();
    while (draw >= limit) {
      draw = engine_();
    }
    return static_cast<int>(draw % bound);
  }

  // Two distinct indices of range(n), n >= 2, like rng.sample(range(n), 2).
  std::pair<int, int> distinct_pair(int n) {
    int first = below(n);
    int second = below(n - 1);
    if (second >= first) {
      ++second;
    }
    return {first, second};
  }

  // k distinct indices of range(n), like rng.sample(range(n), k).
  std::vector<int> sample(int n, int k) {
    std::vector<int> pool(n);
    for (int i = 0; i < n; ++i) pool[i] = i;
    for (int i = 0; i < k; ++i) {
      std::swap(pool[i], pool[i + below(n - i)]);
    }
    pool.resize(k);
    return pool;
  }

  template <typename T>
  void shuffle(std::vector<T>& values) {
    for (int i = static_cast<int>(values.size()) - 1; i > 0; --i) {
      std::swap(values[i], values[below(i + 1)]);
    }
  }

 private:
  std::mt19937_64 engine_;
};

// ---------------------------------------------------------------------------
// Rotations (domain/rotations.py)

Dims rotation_dims(const Item& item, int rotation_id) {
  const int* permutation = ROTATIONS[rotation_id];
  return {item.dims[permutation[0]], item.dims[permutation[1]], item.dims[permutation[2]]};
}

// allowed_rotation_ids(): unique by exact dimensions, in declaration order;
// fitting_rotation_ids(): those whose dimensions fit the bin (with EPS, like
// rotations.FIT_EPSILON in Python).
std::vector<int> fitting_rotation_ids(const Item& item, const Instance& instance) {
  std::vector<int> fitting;
  std::vector<Dims> seen;
  for (int rotation_id : item.allowed_spec) {
    Dims dims = rotation_dims(item, rotation_id);
    if (std::find(seen.begin(), seen.end(), dims) != seen.end()) {
      continue;
    }
    seen.push_back(dims);
    if (dims[0] <= instance.bin_l + EPS && dims[1] <= instance.bin_w + EPS && dims[2] <= instance.bin_h + EPS) {
      fitting.push_back(rotation_id);
    }
  }
  return fitting;
}

std::vector<int> ordered_rotation_ids(const Item& item, int preferred) {
  const auto& fitting = item.fitting;
  if (std::find(fitting.begin(), fitting.end(), preferred) == fitting.end()) {
    return fitting;
  }
  std::vector<int> ordered{preferred};
  for (int rotation_id : fitting) {
    if (rotation_id != preferred) {
      ordered.push_back(rotation_id);
    }
  }
  return ordered;
}

// ---------------------------------------------------------------------------
// Geometry and constraints (domain/validation.py)

Placement make_placement(int item_index, int bin_id, const Point& point, int rotation_id, const Dims& dims) {
  return {item_index,
          bin_id,
          point.x,
          point.y,
          point.z,
          rotation_id,
          dims[0],
          dims[1],
          dims[2],
          point.x + dims[0],
          point.y + dims[1],
          point.z + dims[2],
          dims[0] * dims[1] * dims[2]};
}

bool aabb_overlap(const Placement& a, const Placement& b) {
  return !(a.x2 <= b.x + EPS || b.x2 <= a.x + EPS || a.y2 <= b.y + EPS || b.y2 <= a.y + EPS ||
           a.z2 <= b.z + EPS || b.z2 <= a.z + EPS);
}

double aabb_distance(const Placement& a, const Placement& b) {
  double dx = std::max({b.x - a.x2, a.x - b.x2, 0.0});
  double dy = std::max({b.y - a.y2, a.y - b.y2, 0.0});
  double dz = std::max({b.z - a.z2, a.z - b.z2, 0.0});
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool min_distance_satisfied(const Instance& instance, const Placement& a, const Placement& b) {
  if (instance.required_distance.empty() || a.bin_id != b.bin_id) {
    return true;
  }
  const std::size_t n = instance.items.size();
  double required = instance.required_distance[static_cast<std::size_t>(a.item_index) * n + b.item_index];
  if (required <= 0.0) {
    return true;
  }
  return aabb_distance(a, b) + EPS >= required;
}

double support_ratio(const Placement& placement, const std::vector<Placement>& placements, const std::vector<int>& members) {
  if (placement.z <= EPS) {
    return 1.0;
  }
  double base_area = placement.l * placement.w;
  double supported = 0.0;
  for (int member : members) {
    const Placement& other = placements[member];
    if (std::fabs(other.z2 - placement.z) > EPS) {
      continue;
    }
    double x_overlap = std::max(0.0, std::min(placement.x2, other.x2) - std::max(placement.x, other.x));
    double y_overlap = std::max(0.0, std::min(placement.y2, other.y2) - std::max(placement.y, other.y));
    supported += x_overlap * y_overlap;
  }
  return base_area > 0.0 ? std::min(1.0, supported / base_area) : 1.0;
}

bool point_inside_placement(const Point& point, const Placement& placement) {
  return placement.x + EPS < point.x && point.x < placement.x2 - EPS && placement.y + EPS < point.y &&
         point.y < placement.y2 - EPS && placement.z + EPS < point.z && point.z < placement.z2 - EPS;
}

bool fits_inside_bin(const Instance& instance, const Point& point, const Dims& dims) {
  return point.x >= -EPS && point.y >= -EPS && point.z >= -EPS && point.x + dims[0] <= instance.bin_l + EPS &&
         point.y + dims[1] <= instance.bin_w + EPS && point.z + dims[2] <= instance.bin_h + EPS;
}

bool point_in_bin_bounds(const Instance& instance, const Point& point) {
  return -EPS <= point.x && point.x <= instance.bin_l + EPS && -EPS <= point.y &&
         point.y <= instance.bin_w + EPS && -EPS <= point.z && point.z <= instance.bin_h + EPS;
}

// ---------------------------------------------------------------------------
// Extreme-point decoder (decoders/extreme_points.py)

struct BinState {
  std::vector<int> members;  // indices into the placement list, in placement order
  std::vector<Point> points;  // sorted by (z, y, x)
  double load = 0.0;
};

struct Option {
  bool found = false;
  Rank rank{};
  Placement placement{};
};

// Python tuple order: the first unequal component decides.
bool rank_less(const Rank& left, const Rank& right) {
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right[i]) {
      return left[i] < right[i];
    }
  }
  return false;
}

bool zyx_less(const Point& a, const Point& b) {
  if (a.z != b.z) return a.z < b.z;
  if (a.y != b.y) return a.y < b.y;
  return a.x < b.x;
}

bool zyx_greater(const Point& point, const Placement& placement) {
  if (point.z != placement.z) return point.z > placement.z;
  if (point.y != placement.y) return point.y > placement.y;
  return point.x > placement.x;
}

double contact_score(
    const Instance& instance,
    const Placement& placement,
    const std::vector<Placement>& placements,
    const std::vector<int>& members) {
  double score = 0.0;
  if (std::fabs(placement.x) <= EPS) score += placement.w * placement.h;
  if (std::fabs(placement.y) <= EPS) score += placement.l * placement.h;
  if (std::fabs(placement.z) <= EPS) score += placement.l * placement.w;
  if (std::fabs(placement.x2 - instance.bin_l) <= EPS) score += placement.w * placement.h;
  if (std::fabs(placement.y2 - instance.bin_w) <= EPS) score += placement.l * placement.h;
  if (std::fabs(placement.z2 - instance.bin_h) <= EPS) score += placement.l * placement.w;

  for (int member : members) {
    const Placement& other = placements[member];
    double y_overlap = std::max(0.0, std::min(placement.y2, other.y2) - std::max(placement.y, other.y));
    double z_overlap = std::max(0.0, std::min(placement.z2, other.z2) - std::max(placement.z, other.z));
    double x_overlap = std::max(0.0, std::min(placement.x2, other.x2) - std::max(placement.x, other.x));
    if (std::fabs(placement.x2 - other.x) <= EPS || std::fabs(other.x2 - placement.x) <= EPS) {
      score += y_overlap * z_overlap;
    }
    if (std::fabs(placement.y2 - other.y) <= EPS || std::fabs(other.y2 - placement.y) <= EPS) {
      score += x_overlap * z_overlap;
    }
    if (std::fabs(placement.z2 - other.z) <= EPS || std::fabs(other.z2 - placement.z) <= EPS) {
      score += x_overlap * y_overlap;
    }
  }
  return score;
}

// What _rank_placement needs from the bin's current content; it does not
// depend on the candidate, so it is computed once per bin and item.
struct BinAggregate {
  bool empty = true;
  double max_x = 0.0;
  double max_y = 0.0;
  double max_z = 0.0;
  double occupied = 0.0;  // sum(other.volume for other in bin_placements)
};

BinAggregate aggregate_bin(const Instance& instance, const std::vector<Placement>& placements, const BinState& bin) {
  BinAggregate aggregate;
  PySum occupied(instance.compensated_sum);
  for (int member : bin.members) {
    const Placement& other = placements[member];
    if (aggregate.empty) {
      aggregate.max_x = other.x2;
      aggregate.max_y = other.y2;
      aggregate.max_z = other.z2;
      aggregate.empty = false;
    } else {
      aggregate.max_x = std::max(aggregate.max_x, other.x2);
      aggregate.max_y = std::max(aggregate.max_y, other.y2);
      aggregate.max_z = std::max(aggregate.max_z, other.z2);
    }
    occupied.add(other.volume);
  }
  aggregate.occupied = occupied.value();
  return aggregate;
}

Rank rank_placement(
    const Instance& instance,
    const Placement& placement,
    const std::vector<Placement>& placements,
    const std::vector<int>& members,
    const BinAggregate& aggregate) {
  double max_x = aggregate.empty ? placement.x2 : std::max(placement.x2, aggregate.max_x);
  double max_y = aggregate.empty ? placement.y2 : std::max(placement.y2, aggregate.max_y);
  double max_z = aggregate.empty ? placement.z2 : std::max(placement.z2, aggregate.max_z);
  double bounding_volume = max_x * max_y * max_z;
  double occupied = placement.volume + aggregate.occupied;
  double local_waste = std::max(0.0, bounding_volume - occupied);
  double contact = contact_score(instance, placement, placements, members);

  switch (instance.policy) {
    case Policy::ContactThenZ:
      return {-contact, placement.z, placement.y, placement.x, local_waste};
    case Policy::MinWaste:
      return {local_waste, placement.z, placement.y, placement.x, -contact};
    case Policy::MinZYX:
    default:
      return {placement.z, placement.y, placement.x, local_waste, -contact};
  }
}

// _best_option_for_bin. `best` carries the best option across the bins
// scanned so far and only a strictly better rank replaces it, so the first
// minimal option (bin, then point, then rotation order) wins, as in Python.
void scan_bin(
    const Instance& instance,
    const std::vector<Placement>& placements,
    const BinState& bin,
    int bin_id,
    int item_index,
    const std::vector<int>& rotation_ids,
    Option& best) {
  const Item& item = instance.items[item_index];
  const BinAggregate aggregate = aggregate_bin(instance, placements, bin);
  for (const Point& point : bin.points) {
    // Exact pruning for the default policy: the rank starts with (z, y, x)
    // and the points are sorted by it, so once a point is past the best
    // option no later point of this bin can beat it.
    if (instance.policy == Policy::MinZYX && best.found && zyx_greater(point, best.placement)) {
      break;
    }
    for (int rotation_id : rotation_ids) {
      Dims dims = rotation_dims(item, rotation_id);
      if (!fits_inside_bin(instance, point, dims)) {
        continue;
      }
      Placement candidate = make_placement(item_index, bin_id, point, rotation_id, dims);
      bool blocked = false;
      for (int member : bin.members) {
        const Placement& other = placements[member];
        if (aabb_overlap(candidate, other) || !min_distance_satisfied(instance, candidate, other)) {
          blocked = true;
          break;
        }
      }
      if (blocked) {
        continue;
      }
      if (instance.min_support > 0.0 &&
          support_ratio(candidate, placements, bin.members) + EPS < instance.min_support) {
        continue;
      }
      Rank rank = rank_placement(instance, candidate, placements, bin.members, aggregate);
      if (!best.found || rank_less(rank, best.rank)) {
        best.found = true;
        best.rank = rank;
        best.placement = candidate;
      }
    }
  }
}

// _project: slide the point along the negative axis until it touches an item
// of the bin or the wall (half-open footprint [min, max) on the other axes).
Point project(const Point& point, int axis, const std::vector<Placement>& placements, const std::vector<int>& members) {
  const double coords[3] = {point.x, point.y, point.z};
  double stop = 0.0;
  for (int member : members) {
    const Placement& other = placements[member];
    const double low[3] = {other.x, other.y, other.z};
    const double high[3] = {other.x2, other.y2, other.z2};
    bool over = true;
    for (int a = 0; a < 3; ++a) {
      if (a != axis && !(low[a] - EPS <= coords[a] && coords[a] < high[a] - EPS)) {
        over = false;
        break;
      }
    }
    if (over && high[axis] <= coords[axis] + EPS && high[axis] > stop) {
      stop = high[axis];
    }
  }
  Point projected = point;
  if (axis == 0) {
    projected.x = stop;
  } else if (axis == 1) {
    projected.y = stop;
  } else {
    projected.z = stop;
  }
  return projected;
}

std::vector<Point> update_extreme_points(
    const Instance& instance,
    const std::vector<Point>& points,
    const Placement& placement,
    const std::vector<Placement>& placements,
    const std::vector<int>& members) {
  const Point corner_x{placement.x2, placement.y, placement.z};
  const Point corner_y{placement.x, placement.y2, placement.z};
  const Point corner_z{placement.x, placement.y, placement.z2};
  std::vector<Point> candidates = points;
  candidates.push_back(project(corner_x, 1, placements, members));
  candidates.push_back(project(corner_x, 2, placements, members));
  candidates.push_back(project(corner_y, 0, placements, members));
  candidates.push_back(project(corner_y, 2, placements, members));
  candidates.push_back(project(corner_z, 0, placements, members));
  candidates.push_back(project(corner_z, 1, placements, members));

  std::vector<Point> filtered;
  std::set<Dims> seen;
  for (const Point& point : candidates) {
    if (!point_in_bin_bounds(instance, point)) {
      continue;
    }
    bool inside = false;
    for (int member : members) {
      if (point_inside_placement(point, placements[member])) {
        inside = true;
        break;
      }
    }
    if (inside) {
      continue;
    }
    // round(c * 1_000_000): nearbyint rounds half to even like Python.
    Dims key{std::nearbyint(point.x * 1000000.0), std::nearbyint(point.y * 1000000.0),
             std::nearbyint(point.z * 1000000.0)};
    if (!seen.insert(key).second) {
      continue;
    }
    filtered.push_back(point);
  }

  std::stable_sort(filtered.begin(), filtered.end(), zyx_less);
  const std::size_t limit = static_cast<std::size_t>(std::max<std::size_t>(1, instance.items.size() * 8));
  if (filtered.size() > limit) {
    filtered.resize(limit);
  }
  return filtered;
}

// compute_bin_metrics + compute_fragmentation + score_solution.
void score_decoded(const Instance& instance, Decoded& decoded) {
  int max_bin = -1;
  for (const auto& placement : decoded.placements) {
    max_bin = std::max(max_bin, placement.bin_id);
  }
  const std::size_t bin_count = static_cast<std::size_t>(max_bin + 1);
  std::vector<PySum> occupied(bin_count, PySum(instance.compensated_sum));
  std::vector<Dims> max_corner(bin_count, Dims{0.0, 0.0, 0.0});
  std::vector<bool> used(bin_count, false);
  for (const auto& placement : decoded.placements) {
    const std::size_t bin = static_cast<std::size_t>(placement.bin_id);
    occupied[bin].add(placement.volume);
    max_corner[bin][0] = std::max(max_corner[bin][0], placement.x2);
    max_corner[bin][1] = std::max(max_corner[bin][1], placement.y2);
    max_corner[bin][2] = std::max(max_corner[bin][2], placement.z2);
    used[bin] = true;
  }

  int bins = 0;
  PySum squared_utilization(instance.compensated_sum);
  PySum compactness_sum(instance.compensated_sum);
  for (std::size_t bin = 0; bin < bin_count; ++bin) {
    if (!used[bin]) {
      continue;
    }
    ++bins;
    double bin_occupied = occupied[bin].value();
    double utilization = instance.bin_volume != 0.0 ? bin_occupied / instance.bin_volume : 0.0;
    double bbox_volume = std::max(max_corner[bin][0] * max_corner[bin][1] * max_corner[bin][2], 1e-9);
    double compactness = std::min(1.0, bin_occupied / bbox_volume);
    squared_utilization.add(py_square(utilization));
    compactness_sum.add(compactness);
  }

  decoded.bins = bins;
  decoded.fragmentation = bins > 0 ? 1.0 - squared_utilization.value() / bins : 1.0;
  decoded.compactness_mean = bins > 0 ? compactness_sum.value() / bins : 0.0;
  decoded.objective = static_cast<double>(bins) + decoded.fragmentation;
  decoded.score = {static_cast<double>(bins), decoded.fragmentation, -decoded.compactness_mean};
}

Decoded decode(const Instance& instance, const Candidate& candidate) {
  Decoded decoded;
  decoded.candidate = candidate;
  std::vector<BinState> bins(1);
  bins[0].points.push_back({0.0, 0.0, 0.0});

  for (int item_index : candidate.order) {
    const Item& item = instance.items[item_index];
    int preferred = item_index < static_cast<int>(candidate.rotations.size()) ? candidate.rotations[item_index] : 0;
    std::vector<int> rotation_ids = ordered_rotation_ids(item, preferred);
    if (rotation_ids.empty()) {
      decoded.unplaced.push_back({item_index, "no_fitting_rotation"});
      continue;
    }

    // First fit: the first open bin (in order) that takes the item.
    Option best;
    for (int bin_id = 0; bin_id < static_cast<int>(bins.size()); ++bin_id) {
      if (instance.has_max_weight && bins[bin_id].load + item.weight > instance.max_weight + EPS) {
        continue;
      }
      scan_bin(instance, decoded.placements, bins[bin_id], bin_id, item_index, rotation_ids, best);
      if (best.found) {
        break;
      }
    }
    if (!best.found) {
      // No existing bin fits: open one (it stays open even if the item fails).
      bins.emplace_back();
      bins.back().points.push_back({0.0, 0.0, 0.0});
      int bin_id = static_cast<int>(bins.size()) - 1;
      scan_bin(instance, decoded.placements, bins[bin_id], bin_id, item_index, rotation_ids, best);
    }
    if (!best.found) {
      decoded.unplaced.push_back({item_index, "could_not_be_placed"});
      continue;
    }

    const Placement placed = best.placement;
    decoded.placements.push_back(placed);
    BinState& bin = bins[placed.bin_id];
    bin.members.push_back(static_cast<int>(decoded.placements.size()) - 1);
    bin.load += item.weight;
    bin.points = update_extreme_points(instance, bin.points, placed, decoded.placements, bin.members);
  }

  score_decoded(instance, decoded);
  return decoded;
}

// ---------------------------------------------------------------------------
// Initial pool (constructors/pool.py)

std::vector<int> default_rotation_vector(const Instance& instance) {
  std::vector<int> rotations;
  for (const auto& item : instance.items) {
    rotations.push_back(item.fitting.empty() ? 0 : item.fitting.front());
  }
  return rotations;
}

// sorted(range(n), key=key, reverse=True): stable, ties keep index order.
template <typename Key>
std::vector<int> descending_order(const Instance& instance, Key key) {
  std::vector<int> order(instance.items.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
    return key(instance.items[left]) > key(instance.items[right]);
  });
  return order;
}

std::vector<Candidate> build_initial_pool(const Instance& instance, std::uint64_t seed, int pool_size) {
  Rng rng(seed);
  const auto rotations = default_rotation_vector(instance);
  std::vector<Candidate> pool = {
      {descending_order(instance, [](const Item& item) { return item.volume; }), rotations},
      {descending_order(instance, [](const Item& item) { return std::max({item.dims[0], item.dims[1], item.dims[2]}); }),
       rotations},
      {descending_order(instance,
                        [](const Item& item) {
                          const auto& d = item.dims;
                          return std::max({d[0] * d[1], d[0] * d[2], d[1] * d[2]});
                        }),
       rotations},
  };
  while (static_cast<int>(pool.size()) < pool_size) {
    Candidate random_candidate;
    random_candidate.order.resize(instance.items.size());
    for (std::size_t i = 0; i < instance.items.size(); ++i) random_candidate.order[i] = static_cast<int>(i);
    rng.shuffle(random_candidate.order);
    for (const auto& item : instance.items) {
      const auto& fitting = item.fitting;
      random_candidate.rotations.push_back(
          fitting.empty() ? 0 : fitting[rng.below(static_cast<int>(fitting.size()))]);
    }
    pool.push_back(std::move(random_candidate));
  }

  std::vector<Candidate> unique;
  std::set<std::pair<std::vector<int>, std::vector<int>>> seen;
  for (auto& candidate : pool) {
    if (seen.insert({candidate.order, candidate.rotations}).second) {
      unique.push_back(std::move(candidate));
    }
  }
  // At most pool_size starts (constructors/pool.py does the same).
  if (static_cast<int>(unique.size()) > std::max(1, pool_size)) unique.resize(std::max(1, pool_size));
  return unique;
}

// ---------------------------------------------------------------------------
// Neighbourhood operators (simulated_annealing/operators.py)

void op_swap(Candidate& candidate, Rng& rng) {
  auto& order = candidate.order;
  if (order.size() < 2) return;
  auto [left, right] = rng.distinct_pair(static_cast<int>(order.size()));
  std::swap(order[left], order[right]);
}

void op_relocate(Candidate& candidate, Rng& rng) {
  auto& order = candidate.order;
  if (order.size() < 2) return;
  auto [source, target] = rng.distinct_pair(static_cast<int>(order.size()));
  int item = order[source];
  order.erase(order.begin() + source);
  order.insert(order.begin() + target, item);
}

void op_reverse(Candidate& candidate, Rng& rng) {
  auto& order = candidate.order;
  if (order.size() < 3) return;
  auto [left, right] = rng.distinct_pair(static_cast<int>(order.size()));
  if (left > right) std::swap(left, right);
  std::reverse(order.begin() + left, order.begin() + right + 1);
}

void op_rotate(Candidate& candidate, const Instance& instance, Rng& rng) {
  auto& order = candidate.order;
  if (order.empty()) return;
  int item_index = order[rng.below(static_cast<int>(order.size()))];
  const auto& fitting = instance.items[item_index].fitting;
  if (fitting.size() <= 1) return;
  int current =
      item_index < static_cast<int>(candidate.rotations.size()) ? candidate.rotations[item_index] : fitting[0];
  if (candidate.rotations.size() < instance.items.size()) {
    // A missing entry decodes as preferred rotation 0, same as an explicit 0.
    candidate.rotations.resize(instance.items.size(), 0);
  }
  std::vector<int> alternatives;
  for (int rotation_id : fitting) {
    if (rotation_id != current) alternatives.push_back(rotation_id);
  }
  candidate.rotations[item_index] = alternatives[rng.below(static_cast<int>(alternatives.size()))];
}

void op_ruin_recreate(Candidate& candidate, const Config& config, Rng& rng) {
  auto& order = candidate.order;
  if (order.size() < 3) {
    op_swap(candidate, rng);
    return;
  }
  const int n = static_cast<int>(order.size());
  const int ruin_size = std::min(config.ruin_size, std::max(1, n / 2));
  std::vector<int> positions = rng.sample(n, ruin_size);
  std::sort(positions.rbegin(), positions.rend());
  std::vector<int> removed;
  for (int position : positions) {
    removed.push_back(order[position]);
    order.erase(order.begin() + position);
  }
  rng.shuffle(removed);
  for (int item : removed) {
    order.insert(order.begin() + rng.below(static_cast<int>(order.size()) + 1), item);
  }
}

Candidate propose_neighbor(const Candidate& current, const Instance& instance, const Config& config, Rng& rng) {
  double total = 0.0;
  for (const auto& entry : config.operators) total += entry.second;
  double threshold = rng.random() * total;
  double cumulative = 0.0;
  const std::string* name = &config.operators.front().first;
  for (const auto& entry : config.operators) {
    cumulative += entry.second;
    if (threshold <= cumulative) {
      name = &entry.first;
      break;
    }
  }

  Candidate neighbor = current;
  if (*name == "relocate") {
    op_relocate(neighbor, rng);
  } else if (*name == "reverse") {
    op_reverse(neighbor, rng);
  } else if (*name == "rotate") {
    op_rotate(neighbor, instance, rng);
  } else if (*name == "ruin_recreate") {
    op_ruin_recreate(neighbor, config, rng);
  } else {
    op_swap(neighbor, rng);  // "swap" and, as in Python, any unknown name
  }
  return neighbor;
}

// ---------------------------------------------------------------------------
// Simulated annealing (simulated_annealing/solver.py, schedule.py)

bool score_less(const Score& left, const Score& right) {
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right[i]) {
      return left[i] < right[i];
    }
  }
  return false;
}

double acceptance_probability(double delta, double temperature) {
  if (delta <= 0.0) return 1.0;
  if (temperature <= 0.0) return 0.0;
  return std::exp(-delta / temperature);
}

double seconds_between(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

struct StartOutcome {
  Decoded best;
  int evaluations = 0;
  std::uint64_t seed = 0;
  double initial_temperature = 0.0;
  double stop_temperature = 0.0;
  int calibration_evaluations = 0;
  int uphill_samples = 0;
  bool calibration_fallback = false;
  int reheats = 0;
  std::string stop_reason = "budget";
};

StartOutcome run_single_start(
    const Instance& instance,
    const Config& config,
    int max_evaluations,
    double timeout_seconds,
    std::uint64_t seed,
    const Candidate& initial_candidate,
    double elapsed_offset,
    int evaluation_offset,
    std::vector<ConvergencePoint>& convergence) {
  Rng rng(seed);
  auto started = Clock::now();
  auto elapsed = [&]() { return seconds_between(started, Clock::now()); };

  StartOutcome outcome;
  outcome.seed = seed;
  Decoded current = decode(instance, initial_candidate);
  int evaluations = 1;
  // A start already at the lower bound is optimal: no calibration decodes
  // (the fallback T0 is still computed) and the start ends right below.
  const bool at_lower_bound = current.bins <= config.lower_bound;

  // calibrate_initial_temperature: random walk of single transitions until
  // `temperature_calibration_samples` usable deltas (uphill, same bin count)
  // are collected, within CALIBRATION_MAX_FACTOR times that many steps and the
  // remaining budget. Every step is an evaluation.
  Decoded best = current;
  double temperature = 0.0;
  if (config.initial_temperature.has_value()) {
    temperature = *config.initial_temperature;
  } else {
    const int wanted = config.temperature_calibration_samples;
    const int max_samples = at_lower_bound ? 0 : max_evaluations - evaluations;
    const int max_steps = std::min(wanted * CALIBRATION_MAX_FACTOR, std::max(0, max_samples));
    PySum uphill_sum(instance.compensated_sum);
    int uphill_count = 0;
    Decoded previous = current;
    while (outcome.calibration_evaluations < max_steps && uphill_count < wanted) {
      if (elapsed() >= timeout_seconds) break;
      Decoded sampled = decode(instance, propose_neighbor(previous.candidate, instance, config, rng));
      ++evaluations;
      ++outcome.calibration_evaluations;
      double delta = sampled.objective - previous.objective;
      if (delta > 0.0 && sampled.bins == previous.bins) {
        uphill_sum.add(delta);
        ++uphill_count;
      }
      if (score_less(sampled.score, best.score)) {
        best = sampled;
      }
      previous = std::move(sampled);
    }
    outcome.uphill_samples = uphill_count;
    if (uphill_count == 0) {
      // Within-level deltas scale like 1/bins (r is a mean over bins).
      temperature = FALLBACK_TEMPERATURE / std::max(1, current.bins);
      outcome.calibration_fallback = true;
    } else {
      double mean_delta = uphill_sum.value() / uphill_count;
      temperature = std::max(1e-9, -mean_delta / std::log(config.target_initial_acceptance));
    }
  }
  outcome.initial_temperature = temperature;

  int accepted_since_improvement = 0;
  int reheats = 0;
  convergence.push_back({evaluation_offset + evaluations, elapsed_offset, best.bins, best.objective, true, temperature,
                         current.bins, current.objective, best.bins, best.objective, 0.0, true});

  auto finish = [&](const std::string& reason) {
    outcome.best = std::move(best);
    outcome.evaluations = evaluations;
    outcome.reheats = reheats;
    outcome.stop_reason = reason;
    return outcome;
  };
  if (best.bins <= config.lower_bound) {
    // Optimal already (start or calibration walk reached the bound).
    return finish("lower_bound");
  }

  const int sample_rate = std::max(1, config.convergence_sample_rate);
  // Relative stop temperature, fixed once from T0 (reheating does not move it).
  const double stop_temperature = std::max(config.min_temperature, config.min_temperature_ratio * temperature);
  outcome.stop_temperature = stop_temperature;
  while (evaluations < max_evaluations && temperature >= stop_temperature && elapsed() < timeout_seconds) {
    int level_iterations = std::min(config.iterations_per_temperature, max_evaluations - evaluations);
    for (int iteration = 0; iteration < level_iterations; ++iteration) {
      if (elapsed() >= timeout_seconds) break;
      Decoded neighbor = decode(instance, propose_neighbor(current.candidate, instance, config, rng));
      ++evaluations;

      double delta = neighbor.objective - current.objective;
      bool accepted = rng.random() <= acceptance_probability(delta, temperature);
      bool improved = score_less(neighbor.score, best.score);
      if (improved) {
        best = neighbor;
        accepted_since_improvement = 0;
      }
      if (accepted) {
        current = std::move(neighbor);
        if (!improved) ++accepted_since_improvement;
      }
      if (evaluations % sample_rate == 0 || improved) {
        convergence.push_back({evaluation_offset + evaluations, elapsed_offset + elapsed(), best.bins, best.objective,
                               accepted, temperature, current.bins, current.objective, best.bins, best.objective,
                               delta, improved});
      }

      if (accepted_since_improvement >= config.stagnation_limit) {
        if (config.reheating_factor > 1.0 && reheats < 2) {
          temperature *= config.reheating_factor;
          accepted_since_improvement = 0;
          ++reheats;
        } else {
          return finish("stagnation");
        }
      }
      if (best.bins <= config.lower_bound) {
        return finish("lower_bound");
      }
    }
    temperature *= config.cooling_rate;
  }

  convergence.push_back({evaluation_offset + evaluations, elapsed_offset + elapsed(), best.bins, best.objective, true,
                         temperature, current.bins, current.objective, best.bins, best.objective, 0.0, false});
  std::string reason = "budget";
  if (evaluations < max_evaluations) {
    reason = temperature < stop_temperature ? "min_temperature" : "time_limit";
  }
  return finish(reason);
}

// ---------------------------------------------------------------------------
// Protocol parsing

class TokenReader {
 public:
  explicit TokenReader(std::istream& input) {
    std::string token;
    while (input >> token) tokens_.push_back(token);
  }

  bool done() const { return position_ >= tokens_.size(); }

  std::string next(const std::string& what) {
    if (done()) throw ProtocolError("unexpected end of input while reading " + what);
    return tokens_[position_++];
  }

  double next_double(const std::string& what) {
    std::string token = next(what);
    char* end = nullptr;
    double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || !std::isfinite(value)) {
      throw ProtocolError("invalid number for " + what + ": " + token);
    }
    return value;
  }

  std::optional<double> next_optional_double(const std::string& what) {
    if (!done() && tokens_[position_] == "none") {
      ++position_;
      return std::nullopt;
    }
    return next_double(what);
  }

  long long next_integer(const std::string& what) {
    std::string token = next(what);
    char* end = nullptr;
    long long value = std::strtoll(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') throw ProtocolError("invalid integer for " + what + ": " + token);
    return value;
  }

  int next_int(const std::string& what, long long minimum) {
    long long value = next_integer(what);
    if (value < minimum || value > std::numeric_limits<int>::max()) {
      throw ProtocolError(what + " out of range: " + std::to_string(value));
    }
    return static_cast<int>(value);
  }

  std::uint64_t next_uint64(const std::string& what) {
    std::string token = next(what);
    char* end = nullptr;
    unsigned long long value = std::strtoull(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0' || token[0] == '-') {
      throw ProtocolError("invalid unsigned integer for " + what + ": " + token);
    }
    return static_cast<std::uint64_t>(value);
  }

 private:
  std::vector<std::string> tokens_;
  std::size_t position_ = 0;
};

struct Request {
  std::string mode;
  Instance instance;
  Config config;
  bool items_read = false;
  std::vector<Candidate> initial_pool;
  bool has_candidate = false;
  Candidate candidate;
};

std::vector<int> read_index_list(TokenReader& reader, int count, int n, const std::string& what) {
  std::vector<int> values;
  for (int i = 0; i < count; ++i) {
    int value = reader.next_int(what, 0);
    if (value >= n) throw ProtocolError(what + " out of range: " + std::to_string(value));
    values.push_back(value);
  }
  return values;
}

bool read_config_field(TokenReader& reader, const std::string& key, Config& config) {
  if (key == "seed") config.seed = reader.next_uint64(key);
  else if (key == "max_evaluations") config.max_evaluations = reader.next_int(key, 1);
  else if (key == "time_limit_seconds") config.time_limit_seconds = reader.next_double(key);
  else if (key == "iterations_per_temperature") config.iterations_per_temperature = reader.next_int(key, 1);
  else if (key == "stagnation_limit") config.stagnation_limit = reader.next_int(key, 1);
  else if (key == "ruin_size") config.ruin_size = reader.next_int(key, 1);
  else if (key == "initial_temperature") config.initial_temperature = reader.next_optional_double(key);
  else if (key == "target_initial_acceptance") config.target_initial_acceptance = reader.next_double(key);
  else if (key == "temperature_calibration_samples") config.temperature_calibration_samples = reader.next_int(key, 0);
  else if (key == "cooling_rate") config.cooling_rate = reader.next_double(key);
  else if (key == "min_temperature") config.min_temperature = reader.next_double(key);
  else if (key == "min_temperature_ratio") config.min_temperature_ratio = reader.next_double(key);
  else if (key == "reheating_factor") config.reheating_factor = reader.next_double(key);
  else if (key == "multi_start_count") config.multi_start_count = reader.next_int(key, 1);
  else if (key == "convergence_sample_rate") config.convergence_sample_rate = reader.next_int(key, 1);
  else if (key == "lower_bound") config.lower_bound = reader.next_int(key, 0);
  else if (key == "operators") {
    int count = reader.next_int("operator count", 1);
    config.operators.clear();
    double total = 0.0;
    for (int i = 0; i < count; ++i) {
      std::string name = reader.next("operator name");
      double probability = reader.next_double("operator probability");
      if (probability < 0.0) throw ProtocolError("operator probabilities must be non-negative");
      total += probability;
      config.operators.emplace_back(name, probability);
    }
    if (total <= 0.0) throw ProtocolError("operator probabilities must have positive total mass");
  } else {
    return false;
  }
  return true;
}

void finalize_instance(Instance& instance) {
  instance.bin_volume = instance.bin_l * instance.bin_w * instance.bin_h;
  for (auto& item : instance.items) {
    item.volume = item.dims[0] * item.dims[1] * item.dims[2];
    item.fitting = fitting_rotation_ids(item, instance);
  }
  if (!instance.min_distance.empty()) {
    const std::size_t n = instance.items.size();
    instance.required_distance.assign(n * n, 0.0);
    std::vector<bool> assigned(n * n, false);
    for (const auto& rule : instance.min_distance) {
      for (auto [a, b] : {std::pair<int, int>{rule.a, rule.b}, std::pair<int, int>{rule.b, rule.a}}) {
        std::size_t cell = static_cast<std::size_t>(a) * n + b;
        if (!assigned[cell]) {  // minimum_distance_required(): first matching rule
          assigned[cell] = true;
          instance.required_distance[cell] = rule.distance;
        }
      }
    }
  }
}

Request read_request(std::istream& input) {
  TokenReader reader(input);
  std::string version = reader.next("protocol version");
  if (version != "v2") {
    throw ProtocolError("unsupported native protocol version '" + version + "' (expected v2)");
  }
  Request request;
  Instance& instance = request.instance;
  std::optional<double> support_ratio_min;
  bool stability_required = false;
  bool bin_read = false;

  auto require_items = [&](const std::string& key) {
    if (!request.items_read) throw ProtocolError("'" + key + "' must come after 'items'");
  };

  while (!reader.done()) {
    std::string key = reader.next("key");
    if (key == "end") break;
    if (key == "mode") {
      request.mode = reader.next(key);
      if (request.mode != "solve" && request.mode != "decode") throw ProtocolError("unknown mode: " + request.mode);
    } else if (key == "sum_mode") {
      std::string mode = reader.next(key);
      if (mode != "neumaier" && mode != "naive") throw ProtocolError("unknown sum_mode: " + mode);
      instance.compensated_sum = mode == "neumaier";
    } else if (key == "policy") {
      std::string policy = reader.next(key);
      // _rank_placement falls back to min_z_y_x for unknown ids.
      instance.policy = policy == "contact_then_z" ? Policy::ContactThenZ
                        : policy == "min_waste"    ? Policy::MinWaste
                                                   : Policy::MinZYX;
    } else if (key == "bin") {
      instance.bin_l = reader.next_double("bin length");
      instance.bin_w = reader.next_double("bin width");
      instance.bin_h = reader.next_double("bin height");
      auto max_weight = reader.next_optional_double("bin max_weight");
      instance.has_max_weight = max_weight.has_value();
      instance.max_weight = max_weight.value_or(0.0);
      if (instance.bin_l <= 0.0 || instance.bin_w <= 0.0 || instance.bin_h <= 0.0) {
        throw ProtocolError("bin dimensions must be positive");
      }
      bin_read = true;
    } else if (key == "support") {
      support_ratio_min = reader.next_optional_double("support_ratio_min");
      stability_required = reader.next_int("stability_required", 0) != 0;
    } else if (key == "items") {
      int n = reader.next_int("item count", 1);
      instance.items.assign(n, Item{});
      for (int i = 0; i < n; ++i) {
        Item& item = instance.items[i];
        for (int axis = 0; axis < 3; ++axis) item.dims[axis] = reader.next_double("item dimension");
        auto weight = reader.next_optional_double("item weight");
        item.has_weight = weight.has_value();
        item.weight = weight.value_or(0.0);
        int rotation_count = reader.next_int("allowed rotation count", 0);
        for (int r = 0; r < rotation_count; ++r) {
          int rotation_id = reader.next_int("rotation id", 0);
          if (rotation_id > 5) throw ProtocolError("rotation ids must be between 0 and 5");
          item.allowed_spec.push_back(rotation_id);
        }
      }
      request.items_read = true;
    } else if (key == "min_distance") {
      require_items(key);
      int count = reader.next_int("min_distance count", 0);
      int n = static_cast<int>(instance.items.size());
      for (int i = 0; i < count; ++i) {
        MinDistanceRule rule{};
        rule.a = read_index_list(reader, 1, n, "min_distance item index")[0];
        rule.b = read_index_list(reader, 1, n, "min_distance item index")[0];
        rule.distance = reader.next_double("min_distance distance");
        instance.min_distance.push_back(rule);
      }
    } else if (key == "initial_pool") {
      require_items(key);
      int count = reader.next_int("initial_pool count", 1);
      int n = static_cast<int>(instance.items.size());
      for (int i = 0; i < count; ++i) {
        Candidate candidate;
        candidate.order = read_index_list(reader, n, n, "initial_pool order");
        for (int r = 0; r < n; ++r) candidate.rotations.push_back(reader.next_int("initial_pool rotation", 0));
        request.initial_pool.push_back(std::move(candidate));
      }
    } else if (key == "candidate") {
      require_items(key);
      int n = static_cast<int>(instance.items.size());
      int order_count = reader.next_int("candidate order count", 0);
      request.candidate.order = read_index_list(reader, order_count, n, "candidate order");
      int rotation_count = reader.next_int("candidate rotation count", 0);
      for (int r = 0; r < rotation_count; ++r) {
        request.candidate.rotations.push_back(static_cast<int>(reader.next_integer("candidate rotation")));
      }
      request.has_candidate = true;
    } else if (!read_config_field(reader, key, request.config)) {
      throw ProtocolError("unknown protocol key: " + key);
    }
  }

  if (request.mode.empty()) throw ProtocolError("missing 'mode'");
  if (!bin_read) throw ProtocolError("missing 'bin'");
  if (!request.items_read) throw ProtocolError("missing 'items'");
  if (request.mode == "decode" && !request.has_candidate) throw ProtocolError("decode mode requires 'candidate'");
  // required_support_ratio(): support_ratio_min wins, else stability -> 1.0.
  instance.min_support = support_ratio_min.has_value() ? *support_ratio_min : (stability_required ? 1.0 : 0.0);
  finalize_instance(instance);
  return request;
}

// validate_problem_instance (the subset that makes a solve meaningless).
std::vector<std::string> validate_instance(const Instance& instance) {
  std::vector<std::string> diagnostics;
  for (std::size_t i = 0; i < instance.items.size(); ++i) {
    const Item& item = instance.items[i];
    if (item.volume <= 0.0) {
      diagnostics.push_back("item index " + std::to_string(i) + " has non-positive volume");
    }
    if (item.fitting.empty()) {
      diagnostics.push_back("item index " + std::to_string(i) +
                            " does not fit into the bin in any allowed rotation");
    }
    if (instance.has_max_weight && item.has_weight && item.weight > instance.max_weight) {
      diagnostics.push_back("item index " + std::to_string(i) + " weighs more than the bin max_weight");
    }
  }
  for (const auto& rule : instance.min_distance) {
    if (rule.distance < 0.0) {
      diagnostics.push_back("minimum distance for item indices " + std::to_string(rule.a) + " and " +
                            std::to_string(rule.b) + " must be non-negative");
    }
  }
  return diagnostics;
}

// ---------------------------------------------------------------------------
// JSON output

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;  // max_digits10: round-trips exactly
  return out.str();
}

std::string json_bool(bool value) {
  return value ? "true" : "false";
}

std::string json_string(const std::string& value) {
  std::string out = "\"";
  for (char c : value) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += ' ';
    } else {
      out += c;
    }
  }
  return out + "\"";
}

void write_strings(std::ostream& out, const std::vector<std::string>& values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) out << ",";
    out << json_string(values[i]);
  }
  out << "]";
}

void write_solution_fields(std::ostream& out, const Decoded& decoded) {
  out << "\"best_found_bins\":" << decoded.bins << ",";
  out << "\"objective_scalar\":" << json_number(decoded.objective) << ",";
  out << "\"fragmentation\":" << json_number(decoded.fragmentation) << ",";
  out << "\"compactness_mean\":" << json_number(decoded.compactness_mean) << ",";
  out << "\"placements\":[";
  for (std::size_t i = 0; i < decoded.placements.size(); ++i) {
    const auto& p = decoded.placements[i];
    if (i > 0) out << ",";
    out << "{\"item_index\":" << p.item_index << ",\"bin_id\":" << p.bin_id << ",\"x\":" << json_number(p.x)
        << ",\"y\":" << json_number(p.y) << ",\"z\":" << json_number(p.z) << ",\"rotation_id\":" << p.rotation_id
        << ",\"placed_length\":" << json_number(p.l) << ",\"placed_width\":" << json_number(p.w)
        << ",\"placed_height\":" << json_number(p.h) << "}";
  }
  out << "],\"unplaced\":[";
  for (std::size_t i = 0; i < decoded.unplaced.size(); ++i) {
    if (i > 0) out << ",";
    out << "{\"item_index\":" << decoded.unplaced[i].item_index << ",\"reason\":" << json_string(decoded.unplaced[i].reason)
        << "}";
  }
  out << "]";
}

std::vector<std::string> unplaced_diagnostics(const Decoded& decoded) {
  std::vector<std::string> diagnostics;
  for (const auto& entry : decoded.unplaced) {
    diagnostics.push_back("item index " + std::to_string(entry.item_index) +
                          (entry.reason == "no_fitting_rotation" ? " has no fitting rotation" : " could not be placed"));
  }
  return diagnostics;
}

int run_decode(const Request& request) {
  Decoded decoded = decode(request.instance, request.candidate);
  std::cout << "{\"status\":\"" << (decoded.unplaced.empty() ? "feasible" : "infeasible") << "\",\"mode\":\"decode\",";
  write_solution_fields(std::cout, decoded);
  std::cout << ",\"diagnostics\":";
  write_strings(std::cout, unplaced_diagnostics(decoded));
  std::cout << "}\n";
  return 0;
}

int run_solve(const Request& request) {
  const Instance& instance = request.instance;
  const Config& config = request.config;
  auto solve_started = Clock::now();

  auto diagnostics = validate_instance(instance);
  if (!diagnostics.empty()) {
    std::cout << "{\"status\":\"infeasible\",\"mode\":\"solve\",\"runtime_seconds\":0,\"evaluations\":0,"
              << "\"best_found_bins\":0,\"placements\":[],\"unplaced\":[],\"diagnostics\":";
    write_strings(std::cout, diagnostics);
    std::cout << ",\"convergence\":[]}\n";
    return 0;
  }

  std::vector<Candidate> pool = request.initial_pool;
  if (pool.empty()) {
    pool = build_initial_pool(instance, config.seed, config.multi_start_count);
  }

  // Sequential multi-start: every start gets an equal share of what is left;
  // evaluations a start leaves unused flow to the later ones.
  std::vector<ConvergencePoint> convergence;
  std::vector<StartOutcome> starts;
  std::optional<Decoded> best;
  int evaluations = 0;
  for (std::size_t start_index = 0; start_index < pool.size(); ++start_index) {
    int remaining_evaluations = config.max_evaluations - evaluations;
    double remaining_time = config.time_limit_seconds - seconds_between(solve_started, Clock::now());
    if (remaining_evaluations <= 0 || remaining_time <= 0.0) break;
    int remaining_starts = static_cast<int>(pool.size() - start_index);
    int start_evaluations = std::max(1, remaining_evaluations / remaining_starts);

    StartOutcome outcome = run_single_start(
        instance, config, start_evaluations, std::max(0.001, remaining_time),
        config.seed + static_cast<std::uint64_t>(start_index) * START_SEED_STRIDE, pool[start_index],
        seconds_between(solve_started, Clock::now()), evaluations, convergence);
    evaluations += outcome.evaluations;
    if (!best.has_value() || score_less(outcome.best.score, best->score)) {
      best = outcome.best;
    }
    starts.push_back(std::move(outcome));
    if (best->bins <= config.lower_bound) break;
  }
  if (!best.has_value()) {
    throw ProtocolError("SA did not evaluate any candidate");
  }

  double runtime = seconds_between(solve_started, Clock::now());
  std::cout << "{\"status\":\"" << (best->unplaced.empty() ? "feasible" : "infeasible") << "\",\"mode\":\"solve\",";
  std::cout << "\"runtime_seconds\":" << json_number(runtime) << ",";
  std::cout << "\"evaluations\":" << evaluations << ",";
  std::cout << "\"lower_bound\":" << config.lower_bound << ",";
  write_solution_fields(std::cout, *best);
  std::cout << ",\"diagnostics\":";
  write_strings(std::cout, unplaced_diagnostics(*best));
  std::cout << ",\"starts\":[";
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const auto& s = starts[i];
    if (i > 0) std::cout << ",";
    std::cout << "{\"seed\":" << s.seed << ",\"evaluations\":" << s.evaluations
              << ",\"initial_temperature\":" << json_number(s.initial_temperature)
              << ",\"stop_temperature\":" << json_number(s.stop_temperature)
              << ",\"calibration_evaluations\":" << s.calibration_evaluations
              << ",\"uphill_samples\":" << s.uphill_samples
              << ",\"calibration_fallback\":" << json_bool(s.calibration_fallback) << ",\"reheats\":" << s.reheats
              << ",\"best_bins\":" << s.best.bins << ",\"stop_reason\":" << json_string(s.stop_reason) << "}";
  }
  std::cout << "],\"convergence\":[";
  for (std::size_t i = 0; i < convergence.size(); ++i) {
    const auto& c = convergence[i];
    if (i > 0) std::cout << ",";
    std::cout << "{\"evaluation\":" << c.evaluation << ",\"elapsed_seconds\":" << json_number(c.elapsed_seconds)
              << ",\"bins_used\":" << c.bins_used << ",\"objective_scalar\":" << json_number(c.objective_scalar)
              << ",\"accepted\":" << json_bool(c.accepted) << ",\"temperature\":" << json_number(c.temperature)
              << ",\"current_bins\":" << c.current_bins
              << ",\"current_objective_scalar\":" << json_number(c.current_objective_scalar)
              << ",\"best_bins\":" << c.best_bins
              << ",\"best_objective_scalar\":" << json_number(c.best_objective_scalar)
              << ",\"delta\":" << json_number(c.delta) << ",\"improved\":" << json_bool(c.improved) << "}";
  }
  std::cout << "]}\n";
  return 0;
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  try {
    Request request = read_request(std::cin);
    return request.mode == "decode" ? run_decode(request) : run_solve(request);
  } catch (const ProtocolError& error) {
    std::cout << "{\"status\":\"error\",\"diagnostics\":[" << json_string(error.what()) << "]}\n";
    return 1;
  }
}
