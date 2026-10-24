import pytest

from battleship_sim.board import (
    cell_name,
    cells_of,
    halo,
    is_legal,
    parse_cell,
    placements,
    render,
)
from battleship_sim.rules import FLEET, Rules, Ship, parse_fleet


def test_official_fleet():
    assert [(s.name, s.length) for s in FLEET] == [
        ("Carrier", 5),
        ("Battleship", 4),
        ("Destroyer", 3),
        ("Submarine", 3),
        ("Patrol Boat", 2),
    ]
    assert Rules().fleet_cells == 17
    assert Rules().cells == 100


@pytest.mark.parametrize("size", [5, 10])
@pytest.mark.parametrize("length", [2, 3, 4, 5])
def test_placement_count(size, length):
    # length-L ship on an n x n grid: n rows x (n - L + 1) starts, both orientations
    assert len(placements(length, size)) == 2 * size * (size - length + 1)
    assert len(set(placements(length, size))) == len(placements(length, size))


def test_one_cell_ship_has_one_orientation():
    assert len(placements(1, 10)) == 100


def test_placements_are_straight_and_in_bounds():
    for mask in placements(5, 10):
        cells = cells_of(mask)
        rows = {c // 10 for c in cells}
        cols = {c % 10 for c in cells}
        assert len(cells) == 5
        assert (len(rows) == 1 and max(cols) - min(cols) == 4) or (
            len(cols) == 1 and max(rows) - min(rows) == 4
        )


def test_cell_names_round_trip():
    assert cell_name(0, 10) == "A1"
    assert cell_name(99, 10) == "J10"
    assert cell_name(34, 10) == "D5"
    for cell in range(100):
        assert parse_cell(cell_name(cell, 10), 10) == cell
    with pytest.raises(ValueError):
        parse_cell("K1", 10)


def test_halo_includes_diagonals_and_respects_edges():
    corner = 1 << parse_cell("A1", 10)
    assert sorted(cell_name(c, 10) for c in cells_of(halo(corner, 10))) == [
        "A1", "A2", "B1", "B2",
    ]
    edge_wrap = 1 << parse_cell("A10", 10)  # must not wrap to B1
    assert parse_cell("B1", 10) not in cells_of(halo(edge_wrap, 10))
    middle = 1 << parse_cell("E5", 10)
    assert len(cells_of(halo(middle, 10))) == 9


def _ship(cells, size=10):
    mask = 0
    for name in cells:
        mask |= 1 << parse_cell(name, size)
    return mask


def test_legality_overlap_and_touching():
    rules = Rules(fleet=(Ship("A", 2), Ship("B", 2)))
    no_touch = Rules(fleet=rules.fleet, touching=False)
    a = _ship(["A1", "A2"])
    overlap = _ship(["A2", "A3"])
    side = _ship(["B1", "B2"])
    diagonal = _ship(["B3", "C3"])
    apart = _ship(["C1", "C2"])
    assert not is_legal((a, overlap), rules)
    assert is_legal((a, side), rules)
    assert not is_legal((a, side), no_touch)
    assert is_legal((a, diagonal), rules)
    assert not is_legal((a, diagonal), no_touch)
    assert is_legal((a, apart), no_touch)
    assert not is_legal((a,), rules)  # wrong number of ships
    assert not is_legal((a, _ship(["C1", "C3"])), rules)  # not a real placement


def test_rules_validation():
    with pytest.raises(ValueError):
        Rules(size=4)  # the carrier fits, but 17 cells do not fit in 16
    with pytest.raises(ValueError):
        Rules(size=3, fleet=(Ship("long", 4),))
    with pytest.raises(ValueError):
        Rules(fleet=(Ship("same", 2), Ship("same", 3)))


def test_parse_fleet():
    assert parse_fleet("5,4,3,3,2") == FLEET
    assert [s.length for s in parse_fleet("4,3,2")] == [4, 3, 2]


def test_render():
    rules = Rules(size=3, fleet=(Ship("Patrol Boat", 2),))
    assert render((_ship(["A1", "A2"], 3),), rules) == "P P .\n. . .\n. . ."
