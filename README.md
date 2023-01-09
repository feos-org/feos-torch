# feos-torch

## Installation
Install with (needs rust compiler)

```
pip install git+https://github.com/feos-org/feos-torch
```

## Basic usage
```python
import torch
from feos_torch import PcSaftPure

# m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab
params = torch.tensor([[1.5, 3.5, 150.0, 0, 0.03, 1500.0]] * 2, dtype=torch.float64, requires_grad=True)
temperature = torch.tensor([250., 300.], dtype=torch.float64)
pcsaft = PcSaftPure(params)
vp = pcsaft.vapor_pressure(temperature)
```
