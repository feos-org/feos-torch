from feos_torch.dual import Dual3
import numpy as np


def test_dual():
    x = Dual3.diff(4.0)
    y = 5
    assert x + x == Dual3(8, 2, 0)
    assert x + y == Dual3(9, 1, 0)
    assert y + x == Dual3(9, 1, 0)
    assert -x == Dual3(-4, -1, 0)
    assert x - x == Dual3(0, 0, 0)
    assert x - y == Dual3(-1, 1, 0)
    assert y - x == Dual3(1, -1, 0)
    assert x * x == Dual3(16, 8, 2)
    assert x * y == Dual3(20, 5, 0)
    assert y * x == Dual3(20, 5, 0)
    assert x.recip() == Dual3(0.25, -1 / 16, 1 / 32)
    assert x / x == Dual3(1, 0, 0)
    assert x / y == Dual3(0.8, 0.2, 0)
    assert y / x == Dual3(1.25, -5 / 16, 5 / 32)
    assert np.log(x) == Dual3(np.log(4), 0.25, -1 / 16)
    assert np.exp(x) == Dual3(np.exp(4), np.exp(4), np.exp(4))
    assert np.sqrt(x) == Dual3(2, 0.25, -1 / 32)
