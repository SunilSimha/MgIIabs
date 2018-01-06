"""
A module to compute dN/dWdl for absorbers.
Specifically to calculate dN/dl
"""
import astropy.units as u
from astropy.constants import G as grav
from astropy.units.astrophys import Mpc, M_sun
from numpy import pi
from compos import const

const.initializecosmo()
H0 = 100*u.km/u.s/u.Mpc #(units of h km/s/Mpc)
rho_cr0 = 3*H0**2/(8*pi*grav)
rho_m0 = const.cosmo['omega_0']*rho_cr0

def _test():
    """
    Just to test random things
    """
    return 0

def d2ndWdl(rew=0.1*u.nm,M_low=None,M_high=None,z=0,**kwargs):
    """
    Computes d^2N/dWdl as over the
    mass range [M_low,M_high]
    """
    from . import halomodel as hmod
    from . import halomassfunc as hmf
    from scipy.integrate import quad
    from scipy.optimize import minimize
    import numpy as np
    from compos import const
    import pdb
    import os

    if M_low is None:
        M_low = hmod.lowest_mass(rew,**kwargs).value/1e12
    if M_high is None:
        M_high =  10*M_low

    const.initializecosmo(z=z)

    rho_m = rho_m0.to(M_sun/Mpc**3).value*(1+z)**3

    integrand = lambda M: hmf.dNdM(M*1e12*M_sun,z=z)*rho_m/(M**2*1e12)*np.pi*hmod.rg(M*1e12*M_sun,z=z).value*hmod.p_rew_given_m(rew,M*1e12*M_sun)[0].value
    try:
        integral = quad(integrand,M_low,M_high)
    except ValueError:
        try:
            M_low = M_low+0.01*M_low
            integral = quad(integrand,M_low,M_high)
        except ValueError:
            if M_low>M_high:
                def 
    return integral