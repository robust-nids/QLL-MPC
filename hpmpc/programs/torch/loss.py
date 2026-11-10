"""Specify qll loss, read by qll_to_hpmpc.py"""

import torch

MARGIN = 0.002 # adding a margin helped training in cleartext, instead of strictly satisfying the property only
WEIGHT = 10000.0

class Carrier:
    def __init__(self, p):
        self.p = p
        
class Mul(Carrier):
    """Multiplicative Reals operators of QLL"""
    def tensor(self, a, b): return a * b
    def par(self, a, b): return a * b
    def disj(self, a, b): return (a**self.p + b**self.p)**(1/self.p)
    def conj(self, a, b): return (a**-self.p + b**-self.p)**(-1/self.p)
    def implies(self, a, b): return self.par(self.dual(a), b)
    def dual(self, a): return 1/a

class Add(Carrier):
    """Additive Reals operators of QLL"""
    def tensor(self, a, b): return a + b
    def par(self, a, b): return a + b
    def disj(self, a, b): return torch.logaddexp(-a * self.p, -b * self.p) / -self.p
    def conj(self, a, b): return torch.logaddexp(a * self.p, b * self.p) / self.p
    def implies(self, a, b): return self.par(self.dual(a), b)
    def dual(self, a): return -a
    def leq(self, a, b): return a - b
    def geq(self, a, b): return b - a

# Napiers Isomorphism from multiplicative domain to additive
def to_add(a): return -torch.log(a)

# Napiers Isomorphism from additive domain to multiplicative
def to_mul(a): return torch.exp(-a)

def data_loss(p, y0, y1, y2, y3, y4, l0, l1, l2, l3, l4):
    """data loss for training on data point"""
    return WEIGHT * ((y0 - l0)**2 + (y1 - l1)**2 + (y2 - l2)**2 + (y3 - l3)**2 + (y4 - l4)**2) / 5

def loss(p, y0, y1, y2, y3, y4):
    """QLL Loss for AcasXu for Property 1"""
    q = Add(p)
    maximal = q.conj(q.conj(q.conj(y1 - y0, y2 - y0), y3-y0), y4 - y0)
    return torch.relu(q.dual(maximal) + MARGIN)