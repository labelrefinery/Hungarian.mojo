"""Tests for the rectangular linear assignment solver.

Run with:

    pixi run test
"""

from std.testing import assert_equal, assert_almost_equal, assert_true

from hungarian import (
    CostMatrix,
    Assignment,
    solve,
    solve_gated,
    UNASSIGNED,
    INFEASIBLE,
)


def matrix(
    rows: Int, cols: Int, var values: List[Float64]
) raises -> CostMatrix:
    return CostMatrix(rows, cols, values^)


# ===----------------------------------------------------------------------===
# Brute force reference
# ===----------------------------------------------------------------------===


def _brute_force(cost: CostMatrix, mut used: List[Bool], row: Int) -> Float64:
    """Optimal cost by exhaustive search over injective row -> column maps."""
    if row == cost.rows:
        return 0.0
    var best = Float64.MAX_FINITE
    for c in range(cost.cols):
        if used[c]:
            continue
        used[c] = True
        var rest = _brute_force(cost, used, row + 1)
        used[c] = False
        if rest < Float64.MAX_FINITE:
            var total = cost.get(row, c) + rest
            if total < best:
                best = total
    return best


def brute_force(cost: CostMatrix) -> Float64:
    var used = List[Bool](length=cost.cols, fill=False)
    return _brute_force(cost, used, 0)


struct Lcg(Movable):
    """Deterministic linear congruential generator, so failures reproduce."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_float(mut self) -> Float64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Float64((self.state >> 33) % 1000) / 10.0


# ===----------------------------------------------------------------------===
# Tests
# ===----------------------------------------------------------------------===


def test_two_by_two() raises:
    var m = matrix(2, 2, [4.0, 1.0, 2.0, 3.0])
    var a = solve(m)
    assert_equal(a.col_for_row[0], 1)
    assert_equal(a.col_for_row[1], 0)
    assert_almost_equal(a.total_cost, 3.0)
    assert_equal(a.num_assigned, 2)


def test_three_by_three() raises:
    # Classic worked example; the greedy row-minimum answer (0, 0, 1) is
    # infeasible, so this catches a solver that is really just greedy.
    var m = matrix(3, 3, [8.0, 4.0, 7.0, 5.0, 2.0, 3.0, 9.0, 4.0, 8.0])
    var a = solve(m)
    assert_almost_equal(a.total_cost, brute_force(m))
    # Every row and column used exactly once.
    var seen = List[Bool](length=3, fill=False)
    for r in range(3):
        var c = a.col_for_row[r]
        assert_true(c != UNASSIGNED)
        assert_true(not seen[c])
        seen[c] = True


def test_wide_matrix() raises:
    # 2 tracks, 4 detections: every track matches, two detections are new.
    var m = matrix(2, 4, [9.0, 1.0, 5.0, 3.0, 2.0, 8.0, 4.0, 7.0])
    var a = solve(m)
    assert_equal(a.num_assigned, 2)
    assert_almost_equal(a.total_cost, brute_force(m))
    var unmatched = 0
    for c in range(4):
        if a.row_for_col[c] == UNASSIGNED:
            unmatched += 1
    assert_equal(unmatched, 2)


def test_tall_matrix() raises:
    # 4 tracks, 2 detections: two tracks go unmatched. Exercises the transpose.
    var m = matrix(4, 2, [9.0, 1.0, 5.0, 3.0, 2.0, 8.0, 4.0, 7.0])
    var a = solve(m)
    assert_equal(a.num_assigned, 2)
    var unmatched = 0
    for r in range(4):
        if a.col_for_row[r] == UNASSIGNED:
            unmatched += 1
    assert_equal(unmatched, 2)
    # Consistency between the two directions.
    for r in range(4):
        var c = a.col_for_row[r]
        if c != UNASSIGNED:
            assert_equal(a.row_for_col[c], r)


def test_maximize() raises:
    var m = matrix(2, 2, [4.0, 1.0, 2.0, 3.0])
    var a = solve(m, maximize=True)
    assert_equal(a.col_for_row[0], 0)
    assert_equal(a.col_for_row[1], 1)
    assert_almost_equal(a.total_cost, 7.0)


def test_empty() raises:
    var a = solve(CostMatrix(0, 5))
    assert_equal(a.num_assigned, 0)
    assert_almost_equal(a.total_cost, 0.0)
    var b = solve(CostMatrix(3, 0))
    assert_equal(b.num_assigned, 0)
    assert_equal(len(b.col_for_row), 3)


def test_shape_mismatch_raises() raises:
    var raised = False
    try:
        var bad = CostMatrix(2, 3, List[Float64](length=5, fill=0.0))
        _ = bad^
    except:
        raised = True
    assert_true(raised)


def test_gate_leaves_far_pairs_unmatched() raises:
    # Track 0 sits near detection 0. Track 1's object has left the scene, so
    # every detection is far away. An ungated solve pairs it anyway.
    var m = matrix(2, 2, [0.5, 40.0, 41.0, 39.0])
    var ungated = solve(m)
    assert_equal(ungated.num_assigned, 2)

    var gated = solve_gated(m, gate=5.0)
    assert_equal(gated.num_assigned, 1)
    assert_equal(gated.col_for_row[0], 0)
    assert_equal(gated.col_for_row[1], UNASSIGNED)
    assert_almost_equal(gated.total_cost, 0.5)


def test_gate_still_optimizes_globally() raises:
    # Both tracks are inside the gate for detection 0, but only track 1 is
    # inside it for detection 1. The globally best gated answer gives
    # detection 0 to track 0 even though track 1 is a closer match for it --
    # a solver that greedily assigns the smallest cell first gets this wrong
    # and ends up with one match instead of two.
    var m = matrix(2, 2, [2.0, 90.0, 1.0, 3.0])
    var a = solve_gated(m, gate=5.0)
    assert_equal(a.num_assigned, 2)
    assert_equal(a.col_for_row[0], 0)
    assert_equal(a.col_for_row[1], 1)
    assert_almost_equal(a.total_cost, 5.0)


def test_gate_rejects_everything() raises:
    var m = matrix(2, 2, [50.0, 60.0, 70.0, 80.0])
    var a = solve_gated(m, gate=1.0)
    assert_equal(a.num_assigned, 0)
    assert_almost_equal(a.total_cost, 0.0)


def test_gated_maximize_uses_gate_as_floor() raises:
    # Scores, not distances: keep pairs at or above the floor.
    var m = matrix(2, 2, [0.9, 0.1, 0.05, 0.8])
    var a = solve_gated(m, gate=0.5, maximize=True)
    assert_equal(a.num_assigned, 2)
    assert_almost_equal(a.total_cost, 1.7)

    var strict = solve_gated(m, gate=0.85, maximize=True)
    assert_equal(strict.num_assigned, 1)
    assert_equal(strict.col_for_row[0], 0)


def test_infeasible_marker_is_avoided() raises:
    # The only way to use every row without an INFEASIBLE cell is the
    # anti-diagonal.
    var m = matrix(
        3,
        3,
        [
            INFEASIBLE,
            INFEASIBLE,
            1.0,
            INFEASIBLE,
            2.0,
            INFEASIBLE,
            3.0,
            INFEASIBLE,
            INFEASIBLE,
        ],
    )
    var a = solve(m)
    assert_equal(a.col_for_row[0], 2)
    assert_equal(a.col_for_row[1], 1)
    assert_equal(a.col_for_row[2], 0)
    assert_almost_equal(a.total_cost, 6.0)


def test_random_against_brute_force() raises:
    """The real test: 200 random matrices, exact optimum both ways."""
    var rng = Lcg(0x5EED)
    for trial in range(200):
        var nr = 1 + trial % 5
        var nc = 1 + (trial // 5) % 6
        if nr > nc:
            continue
        var values = List[Float64]()
        for _ in range(nr * nc):
            values.append(rng.next_float())
        var m = matrix(nr, nc, values^)
        var a = solve(m)
        assert_equal(a.num_assigned, nr)
        assert_almost_equal(a.total_cost, brute_force(m), atol=1e-9)


def test_random_transposed_agrees() raises:
    """A tall matrix must cost the same as its wide transpose."""
    var rng = Lcg(0xC0FFEE)
    for trial in range(60):
        var nr = 2 + trial % 4
        var nc = nr + 1 + trial % 3
        var values = List[Float64]()
        for _ in range(nr * nc):
            values.append(rng.next_float())
        var wide = matrix(nr, nc, values^)
        var tall = wide.transposed()
        var a = solve(wide)
        var b = solve(tall)
        assert_almost_equal(a.total_cost, b.total_cost, atol=1e-9)
        for r in range(nr):
            assert_equal(a.col_for_row[r], b.row_for_col[r])


def main() raises:
    test_two_by_two()
    test_three_by_three()
    test_wide_matrix()
    test_tall_matrix()
    test_maximize()
    test_empty()
    test_shape_mismatch_raises()
    test_gate_leaves_far_pairs_unmatched()
    test_gate_still_optimizes_globally()
    test_gate_rejects_everything()
    test_gated_maximize_uses_gate_as_floor()
    test_infeasible_marker_is_avoided()
    test_random_against_brute_force()
    test_random_transposed_agrees()
    print("hungarian: all tests passed")
