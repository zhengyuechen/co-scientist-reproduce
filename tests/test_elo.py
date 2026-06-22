from cosci.elo import expected_score, update

def test_expected_score_equal_ratings():
    assert abs(expected_score(1200, 1200) - 0.5) < 1e-9

def test_update_equal_ratings_k32():
    w, l = update(1200, 1200, k=32, scale=400)
    assert abs(w - 1216.0) < 1e-6   # 1200 + 32*(1-0.5)
    assert abs(l - 1184.0) < 1e-6   # 1200 + 32*(0-0.5)

def test_update_is_zero_sum():
    w, l = update(1300, 1100, k=32)
    assert abs((w - 1300) + (l - 1100)) < 1e-9  # points conserved
