"""Rectangular linear assignment, in pure Mojo.

The data-association half of a tracking-by-detection auto-labeler. Given a
cost matrix over (track, detection) pairs, find the one-to-one matching that
minimizes total cost.

The algorithm is Jonker-Volgenant: sequential shortest augmenting paths over
the residual graph, with Dijkstra's algorithm on reduced costs and dual
variables carried between augmentations. That is the same formulation SciPy's
`linear_sum_assignment` uses, and it runs in O(n^2 m) for an n x m matrix
rather than the O(n^3) of textbook Kuhn-Munkres on a padded square.

Why the dual variables matter: after each augmentation the potentials `u` and
`v` keep every reduced cost non-negative, so the next shortest-path search is
a plain Dijkstra rather than Bellman-Ford. Dropping them and re-running from
scratch is the single most common way a hand-rolled Hungarian implementation
ends up quadratically slower than it should be.

Two entry points, and the difference is the whole story for tracking:

    `solve`        - a true linear assignment. Every row gets a column
                     (assuming rows <= columns). Use it when a match is always
                     legal, however bad.

    `solve_gated`  - assignment with a distance gate. Pairs costing more than
                     the gate are left unmatched rather than forced together.
                     This is what an offline tracker actually wants: a track
                     whose object left the scene must go unmatched, and a
                     solver that always matches will happily bind it to a
                     detection 40 meters away.

A gated solve is not the same as solving on a matrix with the forbidden
entries deleted -- the optimizer still has to reason about them to know which
*other* pairings are worth breaking, so they are admitted at `INFEASIBLE` cost
and stripped afterwards.
"""


# ===----------------------------------------------------------------------===
# Constants
# ===----------------------------------------------------------------------===

comptime UNASSIGNED = -1
"""Sentinel in `Assignment.col_for_row` / `row_for_col` for "no partner"."""

comptime INFEASIBLE = 1.0e18
"""Cost marking a pair that must not be matched.

Large enough that the optimizer will break any number of real pairings before
accepting one, small enough that summing a matrix full of them stays finite.
Set an entry to this rather than deleting it, so the solver still sees the
shape of the problem.
"""

comptime _INF = Float64.MAX_FINITE
"""Internal sentinel for "no path found yet" in the Dijkstra frontier."""


# ===----------------------------------------------------------------------===
# Cost matrix
# ===----------------------------------------------------------------------===


struct CostMatrix(Movable):
    """A dense rectangular cost matrix, row-major.

    Rows are conventionally tracks and columns detections, but nothing here
    depends on that. Storage is one flat `List[Float64]` rather than a list of
    lists: the inner loop of the solver walks a row contiguously, and that is
    the only access pattern that matters for speed.
    """

    var rows: Int
    var cols: Int
    var data: List[Float64]
    """Row-major, `len(data) == rows * cols`."""

    def __init__(out self, rows: Int, cols: Int, fill: Float64 = 0.0):
        self.rows = rows
        self.cols = cols
        self.data = List[Float64](length=rows * cols, fill=fill)

    def __init__(
        out self, rows: Int, cols: Int, var data: List[Float64]
    ) raises:
        """Adopt an existing row-major buffer.

        Raises if the buffer length does not match the declared shape -- a
        silently mis-shaped cost matrix produces a plausible-looking wrong
        matching, which is far worse than a crash.
        """
        if len(data) != rows * cols:
            raise Error(
                "CostMatrix: expected ",
                rows * cols,
                " entries for ",
                rows,
                "x",
                cols,
                ", got ",
                len(data),
            )
        self.rows = rows
        self.cols = cols
        self.data = data^

    def get(self, r: Int, c: Int) -> Float64:
        return self.data[r * self.cols + c]

    def set(mut self, r: Int, c: Int, value: Float64):
        self.data[r * self.cols + c] = value

    def size(self) -> Int:
        return self.rows * self.cols

    def copy(self) -> CostMatrix:
        var out = CostMatrix(self.rows, self.cols)
        for i in range(len(self.data)):
            out.data[i] = self.data[i]
        return out^

    def transposed(self) -> CostMatrix:
        var out = CostMatrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                out.data[c * self.rows + r] = self.data[r * self.cols + c]
        return out^


# ===----------------------------------------------------------------------===
# Result
# ===----------------------------------------------------------------------===


struct Assignment(Movable):
    """A one-to-one matching between rows and columns.

    Both directions are materialized because a tracker needs both: `col_for_row`
    answers "which detection updates this track", `row_for_col` answers "which
    detections are unclaimed and should spawn new tracks".
    """

    var col_for_row: List[Int]
    """`col_for_row[r]` is the matched column, or `UNASSIGNED`."""
    var row_for_col: List[Int]
    """`row_for_col[c]` is the matched row, or `UNASSIGNED`."""
    var total_cost: Float64
    """Sum of the matched entries of the original matrix."""
    var num_assigned: Int

    def __init__(
        out self,
        var col_for_row: List[Int],
        var row_for_col: List[Int],
        total_cost: Float64,
        num_assigned: Int,
    ):
        self.col_for_row = col_for_row^
        self.row_for_col = row_for_col^
        self.total_cost = total_cost
        self.num_assigned = num_assigned

    def is_row_assigned(self, r: Int) -> Bool:
        return self.col_for_row[r] != UNASSIGNED

    def is_col_assigned(self, c: Int) -> Bool:
        return self.row_for_col[c] != UNASSIGNED


# ===----------------------------------------------------------------------===
# Jonker-Volgenant core
# ===----------------------------------------------------------------------===


def _augmenting_path(
    cost: CostMatrix,
    mut u: List[Float64],
    mut v: List[Float64],
    mut path: List[Int],
    row_for_col: List[Int],
    mut shortest: List[Float64],
    mut visited_row: List[Bool],
    mut visited_col: List[Bool],
    mut remaining: List[Int],
    start_row: Int,
    mut min_val: Float64,
) -> Int:
    """Dijkstra over reduced costs from `start_row` to the nearest free column.

    Returns the sink column, or `UNASSIGNED` if no augmenting path exists.
    `min_val` is written back with the distance to the sink; the caller needs
    it to shift the duals.
    """
    var nc = cost.cols
    min_val = 0.0
    var num_remaining = nc
    for i in range(nc):
        # Filled in reverse so the scan below visits columns in ascending
        # order; ties then resolve toward the lower index, which keeps the
        # output stable across runs.
        remaining[i] = nc - i - 1
        shortest[i] = _INF
        visited_col[i] = False
    for i in range(cost.rows):
        visited_row[i] = False

    var sink = UNASSIGNED
    var row = start_row

    while sink == UNASSIGNED:
        visited_row[row] = True
        var index = UNASSIGNED
        var lowest = _INF

        for it in range(num_remaining):
            var col = remaining[it]
            var reduced = min_val + cost.get(row, col) - u[row] - v[col]
            if reduced < shortest[col]:
                path[col] = row
                shortest[col] = reduced
            # A free column at equal distance wins the tie: reaching it ends
            # the search immediately instead of extending the alternating path.
            if shortest[col] < lowest or (
                shortest[col] == lowest and row_for_col[col] == UNASSIGNED
            ):
                lowest = shortest[col]
                index = it

        min_val = lowest
        if index == UNASSIGNED or min_val == _INF:
            return UNASSIGNED

        var col = remaining[index]
        if row_for_col[col] == UNASSIGNED:
            sink = col
        else:
            row = row_for_col[col]

        visited_col[col] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]

    return sink


def _solve_core(cost: CostMatrix) raises -> List[Int]:
    """Assign every row of a matrix with `rows <= cols`. Returns `col_for_row`.
    """
    var nr = cost.rows
    var nc = cost.cols

    var u = List[Float64](length=nr, fill=0.0)
    var v = List[Float64](length=nc, fill=0.0)
    var shortest = List[Float64](length=nc, fill=0.0)
    var path = List[Int](length=nc, fill=UNASSIGNED)
    var col_for_row = List[Int](length=nr, fill=UNASSIGNED)
    var row_for_col = List[Int](length=nc, fill=UNASSIGNED)
    var visited_row = List[Bool](length=nr, fill=False)
    var visited_col = List[Bool](length=nc, fill=False)
    var remaining = List[Int](length=nc, fill=0)

    for cur_row in range(nr):
        var min_val = 0.0
        var sink = _augmenting_path(
            cost,
            u,
            v,
            path,
            row_for_col,
            shortest,
            visited_row,
            visited_col,
            remaining,
            cur_row,
            min_val,
        )
        if sink == UNASSIGNED:
            raise Error(
                "linear assignment is infeasible: no augmenting path from row ",
                cur_row,
            )

        # Shift the duals so every reduced cost stays non-negative. Without
        # this the next search would need Bellman-Ford.
        u[cur_row] += min_val
        for i in range(nr):
            if visited_row[i] and i != cur_row:
                u[i] += min_val - shortest[col_for_row[i]]
        for j in range(nc):
            if visited_col[j]:
                v[j] -= min_val - shortest[j]

        # Walk the path backwards, flipping matched and unmatched edges.
        while True:
            var row = path[sink]
            row_for_col[sink] = row
            var previous = col_for_row[row]
            col_for_row[row] = sink
            sink = previous
            if row == cur_row:
                break

    return col_for_row^


# ===----------------------------------------------------------------------===
# Public API
# ===----------------------------------------------------------------------===


def solve(cost: CostMatrix, maximize: Bool = False) raises -> Assignment:
    """Minimize (or maximize) the total cost of a one-to-one matching.

    Every row is matched when `rows <= cols`, and every column when
    `cols <= rows`. Raises if no complete matching exists, which can only
    happen when the matrix contains non-finite entries.

    For tracking you almost always want `solve_gated` instead: this function
    will pair a track with an arbitrarily distant detection rather than leave
    it alone.
    """
    if cost.rows == 0 or cost.cols == 0:
        return Assignment(
            List[Int](length=cost.rows, fill=UNASSIGNED),
            List[Int](length=cost.cols, fill=UNASSIGNED),
            0.0,
            0,
        )

    # The core requires rows <= cols; solve the transpose and swap back.
    var flipped = cost.rows > cost.cols
    var work = cost.transposed() if flipped else cost.copy()
    if maximize:
        for i in range(len(work.data)):
            work.data[i] = -work.data[i]

    var primary = _solve_core(work)

    var col_for_row = List[Int](length=cost.rows, fill=UNASSIGNED)
    var row_for_col = List[Int](length=cost.cols, fill=UNASSIGNED)
    var total = 0.0
    var count = 0
    for i in range(len(primary)):
        var j = primary[i]
        if j == UNASSIGNED:
            continue
        var r = j if flipped else i
        var c = i if flipped else j
        col_for_row[r] = c
        row_for_col[c] = r
        total += cost.get(r, c)
        count += 1

    return Assignment(col_for_row^, row_for_col^, total, count)


def solve_gated(
    cost: CostMatrix, gate: Float64, maximize: Bool = False
) raises -> Assignment:
    """Assignment where pairs beyond `gate` are left unmatched.

    Entries worse than the gate are admitted to the optimization at
    `INFEASIBLE` cost and stripped from the result afterwards. Admitting them
    matters: the solver has to know a bad pairing exists to judge whether
    breaking some *other* pairing is worth it, and a matrix with those cells
    deleted answers a different question.

    With `maximize`, `gate` is a floor rather than a ceiling -- pairs scoring
    below it are rejected.
    """
    var gated = cost.copy()
    for r in range(cost.rows):
        for c in range(cost.cols):
            var x = cost.get(r, c)
            var reject = x < gate if maximize else x > gate
            if reject:
                gated.set(r, c, -INFEASIBLE if maximize else INFEASIBLE)

    var raw = solve(gated, maximize)

    var col_for_row = List[Int](length=cost.rows, fill=UNASSIGNED)
    var row_for_col = List[Int](length=cost.cols, fill=UNASSIGNED)
    var total = 0.0
    var count = 0
    for r in range(cost.rows):
        var c = raw.col_for_row[r]
        if c == UNASSIGNED:
            continue
        var x = cost.get(r, c)
        var reject = x < gate if maximize else x > gate
        if reject:
            continue
        col_for_row[r] = c
        row_for_col[c] = r
        total += x
        count += 1

    return Assignment(col_for_row^, row_for_col^, total, count)
