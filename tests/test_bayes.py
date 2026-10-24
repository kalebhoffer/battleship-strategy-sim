"""The Bayesian belief must match the exact posterior.

The exact posterior is computed independently of the sampler: list every legal layout
on a small board, replay the shooter's actual shots against each one through the real
game engine (Ocean), and keep the layouts that would have answered every shot exactly
as the shooter was told. Each kept layout is equally likely.
"""

import random

import pytest

from battleship_sim.board import cells_of, is_legal
from battleship_sim.count import enumerate_layouts
from battleship_sim.game import Ocean, solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Feedback, Rules, Ship
from battleship_sim.shooters import make_shooter
from battleship_sim.shooters.bayes import Belief, outcome_entropy


def _exact(rules, shots, told):
    """Exact per-ship position probabilities and cell occupancy given the evidence."""
    keep = []
    for layout in enumerate_layouts(rules):
        ocean = Ocean(layout, rules)
        if all(ocean.fire(c)[0] == t for c, t in zip(shots, told, strict=True)):
            keep.append(layout)
    ships = [{} for _ in rules.fleet]
    occupancy = [0.0] * rules.cells
    for layout in keep:
        for i, p in enumerate(layout):
            ships[i][p] = ships[i].get(p, 0) + 1 / len(keep)
            for c in cells_of(p):
                occupancy[c] += 1 / len(keep)
    return ships, occupancy, len(keep)


def _play_partial(rules, layout, shots_to_play, seed):
    """Fire `shots_to_play` shots with a random shooter and return what was said."""
    ocean = Ocean(layout, rules)
    order = list(range(rules.cells))
    random.Random(seed).shuffle(order)
    shots, told = [], []
    for c in order:
        if len(shots) == shots_to_play or not ocean.afloat:
            break
        result, _ = ocean.fire(c)
        shots.append(c)
        told.append(result)
    return shots, told


SCENARIOS = [
    (Rules(size=5, fleet=(Ship("A", 3), Ship("B", 2), Ship("C", 2)), feedback=fb, touching=t),
     n, seed)
    for fb in Feedback
    for t in (True, False)
    for n, seed in ((6, 1), (10, 2), (14, 3))
]


def _id(x):
    return f"{x.feedback}-{x.touching}" if isinstance(x, Rules) else str(x)


@pytest.mark.parametrize(("rules", "n", "seed"), SCENARIOS, ids=_id)
def test_belief_matches_exact_posterior(rules, n, seed):
    layout = get_placer("uniform")(rules, random.Random(seed))
    shots, told = _play_partial(rules, layout, n, seed)
    exact_ships, exact_occ, consistent = _exact(rules, shots, told)
    assert consistent >= 1  # the true layout is always consistent

    # Hard cases (unnamed hits) mix slowly, so give the sampler room to converge.
    belief = Belief(rules, random.Random(seed), sweeps=20_000)
    for c, t in zip(shots, told, strict=True):
        belief.knowledge.record(c, t)
    marginals = belief.marginals()
    occupancy = belief.occupancy(marginals)
    worst = max(abs(a - b) for a, b in zip(occupancy, exact_occ, strict=True))
    assert worst < 0.03, f"cell occupancy off by {worst:.3f} ({consistent} layouts)"
    for i in range(len(rules.fleet)):
        for p in set(exact_ships[i]) | set(marginals[i]):
            assert marginals[i].get(p, 0) == pytest.approx(exact_ships[i].get(p, 0), abs=0.03)


def test_belief_never_proposes_an_impossible_position():
    rules = Rules(size=5, fleet=(Ship("A", 3), Ship("B", 2)), feedback=Feedback.SINK_ONLY)
    layout = get_placer("uniform")(rules, random.Random(9))
    shots, told = _play_partial(rules, layout, 9, 9)
    exact_ships, _, _ = _exact(rules, shots, told)
    belief = Belief(rules, random.Random(0), sweeps=200)
    for c, t in zip(shots, told, strict=True):
        belief.knowledge.record(c, t)
    for i, w in enumerate(belief.marginals()):
        assert set(w) <= set(exact_ships[i])


def test_outcome_entropy_on_a_known_split():
    # One ship of length 2 on a 1-row strip of 3: positions {0,1} or {1,2}.
    rules = Rules(size=3, fleet=(Ship("A", 2),), feedback=Feedback.HIT_ONLY)
    belief = Belief(rules, random.Random(0), sweeps=500)
    belief.knowledge.record(3, Ocean(((1 << 3) | (1 << 4),), rules).fire(3)[0])
    h = outcome_entropy(belief.marginals(), belief.knowledge, rules)
    assert all(0 <= x <= 1 + 1e-9 for x in h)  # two outcomes: at most 1 bit


@pytest.mark.parametrize("name", ["bayes", "info-gain", "bayes:5"])
@pytest.mark.parametrize("feedback", list(Feedback))
def test_bayesian_shooters_finish(name, feedback):
    for touching in (True, False):
        rules = Rules(feedback=feedback, touching=touching)
        for seed in range(2):
            layout = get_placer("uniform")(rules, random.Random(seed))
            result = solve(layout, rules, make_shooter(name, rules, random.Random(seed)))
            assert result.repeats == 0 and not result.capped
            assert is_legal(layout, rules)


def test_parametric_names():
    rules = Rules()
    assert make_shooter("bayes:3", rules, random.Random(0)).belief.sweeps == 3
    for bad in ("bayes:0", "bayes:x", "probability-map:5"):
        with pytest.raises(ValueError):
            make_shooter(bad, rules, random.Random(0))
