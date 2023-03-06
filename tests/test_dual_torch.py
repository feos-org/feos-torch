from feos_torch.dual_torch import DualTensor
import numpy as np
import torch


def test_dual():
    x = DualTensor.diff(torch.tensor([[1, 2]], dtype=torch.float64))
    y = 5
    z = torch.tensor([[4, 5]], dtype=torch.float64)
    assert x + x == DualTensor(
        torch.tensor([[2, 4]], dtype=torch.float64),
        torch.tensor([[[2, 0], [0, 2]]], dtype=torch.float64),
    )
    assert x + y == DualTensor(
        torch.tensor([[6, 7]], dtype=torch.float64),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float64),
    )
    assert y + x == DualTensor(
        torch.tensor([[6, 7]], dtype=torch.float64),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float64),
    )
    assert x + z == DualTensor(
        torch.tensor([[5, 7]], dtype=torch.float64),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float64),
    )
    assert -x == DualTensor(
        torch.tensor([[-1, -2]], dtype=torch.float64),
        torch.tensor([[[-1, 0], [0, -1]]], dtype=torch.float64),
    )
    assert x - x == DualTensor(
        torch.tensor([[0, 0]], dtype=torch.float64),
        torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float64),
    )
    assert x - y == DualTensor(
        torch.tensor([[-4, -3]], dtype=torch.float64),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float64),
    )
    assert y - x == DualTensor(
        torch.tensor([[4, 3]], dtype=torch.float64),
        torch.tensor([[[-1, 0], [0, -1]]], dtype=torch.float64),
    )
    assert x - z == DualTensor(
        torch.tensor([[-3, -3]], dtype=torch.float64),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float64),
    )
    assert x * x == DualTensor(
        torch.tensor([[1, 4]], dtype=torch.float64),
        torch.tensor([[[2, 0], [0, 4]]], dtype=torch.float64),
    )
    assert x * y == DualTensor(
        torch.tensor([[5, 10]], dtype=torch.float64),
        torch.tensor([[[5, 0], [0, 5]]], dtype=torch.float64),
    )
    assert y * x == DualTensor(
        torch.tensor([[5, 10]], dtype=torch.float64),
        torch.tensor([[[5, 0], [0, 5]]], dtype=torch.float64),
    )
    assert x * z == DualTensor(
        torch.tensor([[4, 10]], dtype=torch.float64),
        torch.tensor([[[4, 0], [0, 5]]], dtype=torch.float64),
    )
    assert x.recip() == DualTensor(
        torch.tensor([[1, 0.5]], dtype=torch.float64),
        torch.tensor([[[-1, 0], [0, -0.25]]], dtype=torch.float64),
    )
    assert x / x == DualTensor(
        torch.tensor([[1, 1]], dtype=torch.float64),
        torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float64),
    )
    assert x / y == DualTensor(
        torch.tensor([[0.2, 0.4]], dtype=torch.float64),
        torch.tensor([[[0.2, 0], [0, 0.2]]], dtype=torch.float64),
    )
    assert y / x == DualTensor(
        torch.tensor([[5, 2.5]], dtype=torch.float64),
        torch.tensor([[[-5, 0], [0, -1.25]]], dtype=torch.float64),
    )
    assert x / z == DualTensor(
        torch.tensor([[0.25, 0.4]], dtype=torch.float64),
        torch.tensor([[[0.25, 0], [0, 0.2]]], dtype=torch.float64),
    )
    assert x.log() == DualTensor(
        torch.tensor([[0, np.log(2)]], dtype=torch.float64),
        torch.tensor([[[1, 0], [0, 0.5]]], dtype=torch.float64),
    )
    assert x.exp() == DualTensor(
        torch.tensor([[[np.exp(1), np.exp(2)]]], dtype=torch.float64),
        torch.tensor([[[np.exp(1), 0], [0, np.exp(2)]]], dtype=torch.float64),
    )
    assert x.sqrt() == DualTensor(
        torch.tensor([[1, np.sqrt(2)]], dtype=torch.float64),
        torch.tensor([[[0.5, 0], [0, 1 / np.sqrt(8)]]], dtype=torch.float64),
    )
