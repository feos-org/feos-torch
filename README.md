# FeOs-torch - Automatic differentiation of phase equilibria

[![repository](https://img.shields.io/pypi/v/feos-torch)](https://pypi.org/project/feos-torch/)

`FeOs-torch` combines the [`FeOs`](https://github.com/feos-org/feos) thermodynamics engine with the machine learning/automatic differentiation framework [PyTorch](https://pytorch.org/). 

```python
import torch
from feos_torch import PcSaftPure

# define PC-SAFT parameters
# m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb
params = torch.tensor([1.5, 3.5, 250.0, 0, 0.03, 1500.0, 1, 1], dtype=torch.float64, requires_grad=True)
pcsaft = PcSaftPure(params.repeat(5, 1))

# evaluate vapor pressures (in Pa)
temperature = torch.tensor([250., 300., 350., 400., 450.], dtype=torch.float64)
_, vp = pcsaft.vapor_pressure(temperature)
print(vp)

# determine the derivatives of the first vapor pressure w.r.t. PC-SAFT parameters
vp[0].backward()
print(params.grad)
```
```terminal
tensor([  20693.5960,  216164.6184, 1049770.6187, 3281855.9640, 7875531.7021],
       dtype=torch.float64, grad_fn=<MulBackward0>)
tensor([-6.7923e+04, -1.7737e+04, -7.0413e+02,  0.0000e+00, -5.7458e+05,
        -6.9122e+01, -3.6892e+04, -3.6892e+04], dtype=torch.float64)

```

## Models

The following models and properties are currently implemented in `FeOs-torch`

|model|vapor pressure|liquid density|equilibrium liquid density|bubble point pressure|dew point pressure|
|-|-|-|-|-|-|
|PC-SAFT|✓|✓|✓|✓|✓|
|gc-PC-SAFT||||✓|✓|

## Cite us

If you find `FeOs-torch` useful for your own research, consider citing our [publication](https://doi.org/10.1007/s10765-023-03290-3) from which this library resulted.

```
@article{rehner2023mixtures,
  author = {Rehner, Philipp and Bardow, André and Gross, Joachim},
  title = {Modeling Mixtures with PCP-SAFT: Insights from Large-Scale Parametrization and Group-Contribution Method for Binary Interaction Parameters}
  journal = {International Journal of Thermophysics},
  volume = {44},
  number = {12},
  pages = {179},
  year = {2023}
}
```