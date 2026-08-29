# Hungarian.mojo

Rectangular linear assignment, in pure Mojo.

The data-association half of a tracking-by-detection auto-labeler: given a cost
matrix over (track, detection) pairs, find the one-to-one matching that
minimizes total cost. Pair it with [Kalman.mojo](https://github.com/labelrefinery/Kalman.mojo)
for the state-estimation half.

The algorithm is Jonker-Volgenant — sequential shortest augmenting paths, with
Dijkstra on reduced costs and dual variables carried between augmentations.
Same formulation as SciPy's `linear_sum_assignment`, running in O(n²m) on an
n×m matrix rather than the O(n³) of textbook Kuhn-Munkres on a padded square.

## Why the gate matters

There are two entry points and the difference is the whole story for tracking:

- `solve` is a true linear assignment. Every row gets a column. Use it when a
  match is always legal, however bad.
- `solve_gated` leaves pairs beyond a distance gate unmatched. This is what an
  offline tracker actually wants — a track whose object left the scene must go
  unmatched, and a solver that always matches will happily bind it to a
  detection 40 meters away.

A gated solve is *not* the same as solving on a matrix with the forbidden
entries deleted. The optimizer still has to reason about them to know which
*other* pairings are worth breaking, so they are admitted at `INFEASIBLE` cost
and stripped afterwards.

```mojo
from hungarian import CostMatrix, solve_gated, UNASSIGNED

# Rows are tracks, columns are detections. Both tracks are inside the gate for
# detection 0, but only track 1 is inside it for detection 1.
var costs: List[Float64] = [2.0, 90.0, 1.0, 3.0]
var m = CostMatrix(2, 2, costs^)

var a = solve_gated(m, gate=5.0)
# a.col_for_row == [0, 1] -- two matches, total cost 5.0.
# A greedy solver takes the 1.0 cell first and ends up with one match.
```

## API

`src/hungarian/__init__.mojo` is the whole library.

| type / function | role |
| --- | --- |
| `CostMatrix` | dense rectangular costs, flat row-major |
| `Assignment` | the matching: `col_for_row`, `row_for_col`, `total_cost`, `num_assigned` |
| `solve(cost, maximize=False)` | exact linear assignment; raises if infeasible |
| `solve_gated(cost, gate, maximize=False)` | assignment with pairs beyond `gate` left unmatched |
| `UNASSIGNED` | sentinel (`-1`) for "no partner" |
| `INFEASIBLE` | cost marking a pair that must not be matched |

Any orientation works. With `rows <= cols` every row is matched; with
`cols <= rows` every column is. `Assignment` materializes both directions
because a tracker needs both: `col_for_row` answers "which detection updates
this track", `row_for_col` answers "which detections are unclaimed and should
spawn new tracks".

With `maximize=True` the gate becomes a floor rather than a ceiling, so a
similarity score works as naturally as a distance.

## Install

As a [mojoshelf](https://mojoshelf.org) tin, in pixi mode:

```
pixi shelf add hungarian-mojo
```

or as a submodule:

```
shelf add hungarian-mojo      # then: mojo run -I shelf/hungarian-mojo/src ...
```

The registry and conda package name is `hungarian-mojo`, while the Mojo import
name stays `hungarian`:

```mojo
from hungarian import CostMatrix, solve
```

## Tests

```
pixi run test
```

Fourteen cases: hand-checked small matrices, both rectangular orientations,
maximize, gating, and the one that matters — 200 randomly generated matrices
whose optimum is cross-checked against exhaustive search over all injective
row→column maps.

## License

Apache 2.0.
