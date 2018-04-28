"""
A module containing functions from Zhu and Menard's
empirical models.
"""

_g0 = 0.63
_ag = 5.38
_zg = 0.41
_bg = 2.97

_W0 = 0.33
_aw = 1.21
_zw = 2.24
_bw = 2.43

def _g(z):
    return _g0*(1+z)**_ag/(1+(z/_zg)**_bg)

def _wstar(z):
    return _W0*(1+z)**_aw/(1+(z/_zw)**_bw)

def obs_d2ndwdz(w,z):
    from numpy import exp
    return _g(z)*exp(-w/_wstar(z))