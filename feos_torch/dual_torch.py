import torch


class DualTensor:
    def __init__(self, re, eps1, eps2, eps1eps2):
        self.re = re
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps1eps2 = eps1eps2

    def __repr__(self):
        return f"re: {self.re}\neps1: {self.eps1}\neps2: {self.eps2}\neps1eps2: {self.eps1eps2}"

    def __getitem__(self, key):
        return DualTensor(
            self.re[key],
            self.eps1[key + (slice(None, None, None),)],
            self.eps2[key],
            self.eps1eps2[key + (slice(None, None, None),)],
        )

    def __setitem__(self, key, value):
        self.re[key] = value.re
        self.eps1[key + (slice(None, None, None),)] = value.eps1
        self.eps2[key] = value.eps2
        self.eps1eps2[key + (slice(None, None, None),)] = value.eps1eps2

    def __len__(self):
        return self.re.__len__()

    def sum(self, dim, keepdim):
        re = self.re.sum(dim, keepdim=keepdim)
        eps1 = self.eps1.sum(dim, keepdim=keepdim)
        eps2 = self.eps2.sum(dim, keepdim=keepdim)
        eps1eps2 = self.eps1eps2.sum(dim, keepdim=keepdim)
        return DualTensor(re, eps1, eps2, eps1eps2)

    def norm(self):
        return self.re.norm()

    def __eq__(self, other):
        if isinstance(other, DualTensor):
            return (
                torch.all(self.re == other.re)
                and torch.all(self.eps1 == other.eps1)
                and torch.all(self.eps2 == other.eps2)
                and torch.all(self.eps1eps2 == other.eps1eps2)
            )
        return False

    def __lt__(self, other):
        return self.re < other

    def __add__(self, other):
        if isinstance(other, DualTensor):
            return DualTensor(
                self.re + other.re,
                self.eps1 + other.eps1,
                self.eps2 + other.eps2,
                self.eps1eps2 + other.eps1eps2,
            )
        return DualTensor(self.re + other, self.eps1, self.eps2, self.eps1eps2)

    def __neg__(self):
        return DualTensor(-self.re, -self.eps1, -self.eps2, -self.eps1eps2)

    def __sub__(self, other):
        if isinstance(other, DualTensor):
            return DualTensor(
                self.re - other.re,
                self.eps1 - other.eps1,
                self.eps2 - other.eps2,
                self.eps1eps2 - other.eps1eps2,
            )
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, DualTensor):
            re = self.re * other.re
            eps1 = self.re[:, :, None] * other.eps1 + other.re[:, :, None] * self.eps1
            eps2 = self.re * other.eps2 + other.re * self.eps2
            eps1eps2 = (
                self.re[:, :, None] * other.eps1eps2
                + self.eps1 * other.eps2[:, :, None]
                + self.eps2[:, :, None] * other.eps1
                + self.eps1eps2 * other.re[:, :, None]
            )
            return DualTensor(re, eps1, eps2, eps1eps2)

        if isinstance(other, torch.Tensor):
            return DualTensor(
                self.re * other,
                self.eps1 * other[:, :, None],
                self.eps2 * other,
                self.eps1eps2 * other[:, :, None],
            )

        if isinstance(other, (float, int)):
            return DualTensor(
                self.re * other,
                self.eps1 * other,
                self.eps2 * other,
                self.eps1eps2 * other,
            )

    def chain_rule(self, f0, f1, f2):
        re = f0
        eps1 = f1[:, :, None] * self.eps1
        eps2 = f1 * self.eps2
        eps1eps2 = (
            f1[:, :, None] * self.eps1eps2
            + f2[:, :, None] * self.eps1 * self.eps2[:, :, None]
        )
        return DualTensor(re, eps1, eps2, eps1eps2)

    def recip(self):
        rec = 1 / self.re
        rec2 = rec * rec
        return self.chain_rule(rec, -rec2, 2 * rec2 * rec)

    def __truediv__(self, other):
        if isinstance(other, DualTensor):
            return self * other.recip()

        if isinstance(other, torch.Tensor):
            return DualTensor(
                self.re / other,
                self.eps1 / other[:, :, None],
                self.eps2 / other,
                self.eps1eps2 / other[:, :, None],
            )

        if isinstance(other, (float, int)):
            return DualTensor(
                self.re / other,
                self.eps1 / other,
                self.eps2 / other,
                self.eps1eps2 / other,
            )

    def __rtruediv__(self, other):
        return other * self.recip()

    def log(self):
        re = self.re
        rec = 1 / re
        return self.chain_rule(re.log(), rec, -rec * rec)

    def exp(self):
        e = self.re.exp()
        return self.chain_rule(e, e, e)

    def sqrt(self):
        s = self.re.sqrt()
        return self.chain_rule(s, 0.5 / s, -0.25 / s / self.re)


DualTensor.__radd__ = DualTensor.__add__
DualTensor.__rmul__ = DualTensor.__mul__


class Dual2:
    def __init__(self, re, eps1, eps2):
        self.re = re
        self.eps1 = eps1
        self.eps2 = eps2

    def __repr__(self):
        return f"{self.re} + {self.eps1}eps1 +  {self.eps2}eps2"

    def __mul__(self, other):
        if isinstance(other, Dual2):
            re = self.re * other.re
            eps1 = self.re * other.eps1 + self.eps1 * other.re
            eps2 = self.re * other.eps2 + self.eps2 * other.re
            return Dual2(re, eps1, eps2)
        return Dual2(self.re * other, self.eps1 * other, self.eps2 * other)

    def __truediv__(self, other):
        if isinstance(other, Dual2):
            re = self.re / other.re
            eps1 = (self.eps1 * other.re - self.re * other.eps1) / (other.re * other.re)
            eps2 = (self.eps2 * other.re - self.re * other.eps2) / (other.re * other.re)
            return Dual2(re, eps1, eps2)
        return Dual2(self.re / other, self.eps1 / other, self.eps2 / other)

    def __add__(self, other):
        if isinstance(other, Dual2):
            re = self.re + other.re
            eps1 = self.eps1 + other.eps1
            eps2 = self.eps2 + other.eps2
            return Dual2(re, eps1, eps2)
        return Dual2(self.re + other, self.eps1, self.eps2)

    def __sub__(self, other):
        if isinstance(other, Dual2):
            re = self.re - other.re
            eps1 = self.eps1 - other.eps1
            eps2 = self.eps2 - other.eps2
            return Dual2(re, eps1, eps2)
        return Dual2(self.re - other, self.eps1, self.eps2)


Dual2.__radd__ = Dual2.__add__
Dual2.__rmul__ = Dual2.__mul__
