"""
A module to compute dN/dWdl for absorbers.
Specifically to calculate dN/dl
"""
import astropy.units as u
from astropy.constants import G as grav, c
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

def d2ndWdl(rew,z=0.0,growthf=None,spline_interp=None):
    """
    This evaluates the integral for $d^2N/dW_rdl$
    by approximating the integrand in log-normal
    distribution. The integrand in this approximation
    turns out to be the errorfunction simply.
    Parameters
    ----------
    rew: astropy.Quantity
        Rest equivalent width in nanometers
    z: float, optional
        Redshift
    growthf: float, optional
        The growthfactor corresponding to 
        the input redshift. The function
        works faster if supplied.
    spline_interp: function
        The interpolating function for the
        gaussian parameters. If not given,
        it will be computed. Works faster
        if supplied.
    Returns
    -------
    integral: astropy.Quantity
        The integral in units of
        $nm^{-1} h Mpc^{-1}$
    """
    from compos import growthfactor as gf
    from astropy.table import Table
    from numpy import log10, array
    from pkg_resources import resource_filename
    from scipy.interpolate import interp1d
    from scipy.special import ndtr
    from .halomodel import lowest_mass

    #A bit costly. Best to provide
    #these things to speed up calculations
    if growthf is None:
        growthf = gf.growfunc_z(z)
    if spline_interp is None:
        filename = resource_filename('MgIIabs.model','')[:-5]+"data/gauss_params.csv"
        params = Table.read(filename,format="ascii.csv")
        x = params['z']
        y = array([params['A'],params['mu'],params['sigma']])
        spline_interp = interp1d(x,y,kind="cubic")
    A, mu, sigma = spline_interp(z)

    M_low = log10(lowest_mass(rew,low=3,z=z).value/1e12)

    integral = A*ndtr(-(M_low+mu)/sigma)
    return integral/u.nm/u.Mpc

def _dldz(z,spline_interp=None):
    """
    Variation of the comoving radial
    distance as function of redshift
    Cosmology: Planck 13
    Parameters
    ----------
    z: float
        Redshift
    spline_interp: function, optional
        Spline interpolation of gaussian
        parameters.
    Returns
    -------
    dlbydz: astropy.Quantity
        In units of h^-1 Mpc
    """
    from numpy import sqrt
    const.initializecosmo()
    omega_m = const.cosmo['omega_0']
    omega_DE = const.cosmo['omega_q']
    return (c/H0/sqrt(omega_m*(1+z)**3+omega_DE)).to(Mpc)

def dNdz(z,rew_min, rew_max=float('inf'),spline_interp=None):
    """
    Integral of d^2N/dWdl over the specified REW
    window.
    Parameters
    ----------
    z: float
        Redshift
    rew_min: astropy.Quantity
        Minimum rest equivalent width (nm)
    rew_max: astropy.Quantity, optional
        Maximum REW (Inf by default)
    Returns
    -------
    integral: float
    """
    from compos import growthfactor as gf
    from astropy.table import Table
    from numpy import log10, array, isinf
    from pkg_resources import resource_filename
    from scipy.interpolate import interp1d
    from scipy.integrate import quad
    from scipy.special import ndtr
    from .halomodel import lowest_mass

    if spline_interp is None:
        filename = resource_filename('MgIIabs.model','')[:-5]+"data/gauss_params.csv"
        params = Table.read(filename,format="ascii.csv")
        x = params['z']
        y = array([params['A'],params['mu'],params['sigma']])
        spline_interp = interp1d(x,y,kind="cubic")
    A, mu, sigma = spline_interp(z)

    M_low1 = log10(lowest_mass(rew_min,low=3,z=z).value/1e12)
    f = lambda x: ndtr(-x)

    if isinf(rew_max):
        integral = A*_dldz(z)*quad(f,(M_low1+mu)/sigma,float('inf'))[0]
    else:
        M_low2 = log10(lowest_mass(rew_max,low=0,z=z).value/1e12)
        integral = A*_dldz(z).value*quad(f,(M_low1+mu)/sigma,(M_low2+mu)/sigma)[0]
    return integral       

#def d2ndWdl(rew=0.1*u.nm,M_low=None,M_high=4,z=0,growthf=1,**kwargs):
#    """
#    Computes d^2N/dWdl as over the
#    mass range [M_low,M_high]
#    """
#    from . import halomodel as hmod
#    from . import halomassfunc as hmf
#    from scipy.integrate import quad
#    import numpy as np
#
#    if M_low is None:
#        M_low = np.log10(hmod.lowest_mass(rew,z=z,**kwargs).value/1e12)
#    if M_high<M_low:
#        return 0.0
#    const.initializecosmo(z=z)
#
#    rho_m = rho_m0.to(M_sun/Mpc**3).value*(1+z)**3
#
#    def integrand(logM_12):
#        M = 10**logM_12*1e12
#        dndm = hmf.dNdM(M*M_sun,z=z,growthf=growthf)*rho_m/M**2
#        sigmag = np.pi*hmod.rg(M*M_sun,z=z).value**2
#        pWM = hmod.p_rew_given_m(rew,M*M_sun,z=z)[0].value
#        return dndm*sigmag*pWM*M
#    try:
#        integral = np.log(10)*quad(integrand,M_low,M_high)[0]
#        return integral
#    except ValueError:
#        try:
#            M_low = M_low+0.01*M_low
#            integral = quad(integrand,M_low,M_high)
#            return integral
#        except ValueError:
#            #if M_low>M_high:
#            pdb.set_trace()