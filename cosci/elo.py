"""Elo ranking. OURS: K=32, scale=400 (SM specifies only initial rating 1200)."""

INITIAL_ELO = 1200  # SM: initial Elo rating assigned to a hypothesis on tournament entry


def expected_score(r_a: float, r_b: float, scale: int = 400) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))


def update(winner: float, loser: float, k: int = 32, scale: int = 400) -> tuple[float, float]:
    e_w = expected_score(winner, loser, scale)
    e_l = 1.0 - e_w
    new_winner = winner + k * (1.0 - e_w)
    new_loser = loser + k * (0.0 - e_l)
    return new_winner, new_loser
