import random
from utils import inverse_cdf_sample

def random_combination(xs,k):
    """return a k-combination of xs uniformly at random"""
    comb = []
    while len(comb) < k:
        x = random.choice(xs)
        if not x in comb:
            comb.append(x)
    return comb

def rpower_law(alpha=2,M=1000):
    return inverse_cdf_sample(range(1,M+1),[1.0/(i**alpha) for i in range(1,M+1)],normalized=False)
