from torch import Tensor
import numpy as np


class Dual3:
    def __init__(self, re, v1, v2):
        self.re = re
        self.v1 = v1
        self.v2 = v2

    @classmethod
    def diff(cls, re):
        return cls(re, 1, 0)

    def __repr__(self):
        return f"{self.re} + {self.v1}v1 + {self.v2}v2"

    def __eq__(self, other):
        if isinstance(other, Dual3):
            return self.re == other.re and self.v1 == other.v1 and self.v2 == other.v2
        return False

    def __add__(self, other):
        if isinstance(other, Dual3):
            return Dual3(self.re + other.re, self.v1 + other.v1, self.v2 + other.v2)
        return Dual3(self.re + other, self.v1, self.v2)

    def __neg__(self):
        return Dual3(-self.re, -self.v1, -self.v2)

    def __sub__(self, other):
        if isinstance(other, Dual3):
            return Dual3(self.re - other.re, self.v1 - other.v1, self.v2 - other.v2)
        return Dual3(self.re - other, self.v1, self.v2)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, Dual3):
            return Dual3(
                self.re * other.re,
                self.v1 * other.re + self.re * other.v1,
                self.v2 * other.re + 2 * self.v1 * other.v1 + self.re * other.v2,
            )
        return Dual3(self.re * other, self.v1 * other, self.v2 * other)

    def chain_rule(self, f0, f1, f2):
        return Dual3(f0, f1 * self.v1, f2 * self.v1**2 + f1 * self.v2)

    def recip(self):
        rec = 1 / self.re
        return self.chain_rule(rec, -(rec**2), 2 * rec**3)

    def __truediv__(self, other):
        if isinstance(other, Dual3):
            return self * other.recip()
        return Dual3(self.re / other, self.v1 / other, self.v2 / other)

    def __rtruediv__(self, other):
        return other * self.recip()

    def log(self):
        rec = 1 / self.re
        l = self.re.log() if isinstance(self.re, Tensor) else np.log(self.re)
        return self.chain_rule(l, rec, -(rec**2))

    def exp(self):
        e = self.re.exp() if isinstance(self.re, Tensor) else np.exp(self.re)
        return self.chain_rule(e, e, e)

    def sqrt(self):
        s = self.re.sqrt() if isinstance(self.re, Tensor) else np.sqrt(self.re)
        return self.chain_rule(s, 0.5 / s, -0.25 / (s * s * s))


Dual3.__radd__ = Dual3.__add__
Dual3.__rmul__ = Dual3.__mul__
