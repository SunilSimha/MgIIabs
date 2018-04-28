"""
A module to compute dN/dWdl for absorbers.
Specifically to calculate dN/dl
"""
import astropy.units as u
from astropy.constants import G as grav, c
from astropy.units.astrophys import Mpc, M_sun
from astropy.cosmology import Planck13
from numpy import pi, exp, linspace,log10, vectorize, asarray
from compos import const
from hmf import MassFunction

h = Planck13.h
H0 = 100*u.km/u.s/u.Mpc #(units of h km/s/Mpc)
rho_cr0 = 3*H0**2/(8*pi*grav)
rho_m0 = Planck13.Om0*rho_cr0
mass_func = MassFunction(Mmin=6,Mmax=16,dlog10m=0.05)

def _test():
    """
    Just to test random things
    """
    return 0
def d2ndWdl(rew,z=0,M_low=None,M_high=None,**kwargs):
    """
    Computes d^2N/dWdl as over the
    mass range [M_low,M_high]
    """
    from . import halomodel as hmod
    #from . import halomassfunc as hmf
    from scipy.integrate import simps
    import numpy as np
    import pdb
    #pdb.set_trace()
    M_high = 16
    if M_low is None:
        #I won't be considering halos lower than 1e9 h^-1 Msun
        #This is simply because of the for of kappa_g
        try:
            M_low = max(np.log10(hmod.lowest_mass(rew,z=z).value),8)
        except ValueError:
            return 0.0/u.nm/Mpc
    if M_high<M_low:
        return 0.0/u.nm/Mpc
    mass_func.update(z=z)
    massfilter = (mass_func.m>=10**M_low)*(mass_func.m<=10**M_high)
    masses = mass_func.m[massfilter]
    if len(masses)==0:
        return 0.0/u.nm/Mpc
    dndm = mass_func.dndm[massfilter]
    Rg = hmod.rg(masses*M_sun,z=z).value
    p = np.asarray([hmod.p_rew_given_m(rew,M*M_sun,z=z,kappa_m=kwargs['kappa_m']).value for M in masses])
    
    integrand = p*np.pi*Rg**2*dndm*masses
    try:
        integral = simps(integrand,np.log10(masses))*np.log(10)
    except:
        pdb.set_trace()
    return integral/u.nm/Mpc

    

    ##const.initializecosmo(z=z)
    #
    #rho_m = rho_m0.to(M_sun/Mpc**3).value*(1+z)**3
#
    #def integrand(logM_12):
    #    M = 10**logM_12*1e12
    #    dndm = hmf.dNdM(M*M_sun,z=z,growthf=growthf)*rho_m/M**2
    #    sigmag = np.pi*hmod.rg(M*M_sun,z=z).value**2
    #    pWM = hmod.p_rew_given_m(rew,M*M_sun,z=z)[0].value
    #    return dndm*sigmag*pWM*M
    #try:
    #    integral = np.log(10)*quad(integrand,M_low,M_high,points=[-0.67,0.66])[0]
    #    return integral/u.nm/u.Mpc
    #except ValueError:
    #    try:
    #        M_low = M_low+0.01*abs(M_low)
    #        integral = np.log(10)*quad(integrand,M_low,M_high)
    #        return integral/u.nm/u.Mpc
    #    except ValueError:
    #        #if M_low>M_high:
    #        pdb.set_trace()
    #except IntegrationWarning:
    #    pdb.set_trace()

def _dldz(z):
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

def d2ndWdz(rew,z=0.0,growthf=1,spline_interp=None,**kwargs):
    """
    Replaces the comoving distance with redshift
    """
    import pdb
    #pdb.set_trace()
    return (d2ndWdl(rew,z,growthf,kappa_m=kwargs['kappa_m'])*_dldz(z)).to(1/u.nm)

def _expJoin(x0,a,b):
    """
    extrapolates a decaying function as an exponential
    by imposing continuity and differentiability at the
    end point.
    Parameters
    ----------
    x0: float
        Right end point
    a: float
        Function value at x0
    b: float
        Function derivative at x0
    Returns
    -------
    expFunc: function
        An extrapolating function
    """
    B = -b/a
    A = a*exp(b*x0/a)
    expFunc = lambda x: A*exp(-B*x)
    return expFunc
    #If an interpolation function is not provided,
    #The integration is one by brute force.
    #    redshifts = np.linspace(0,2.5)
    #    filename = resource_filename('MgIIabs.count',"")[:-5]+"/data/d2ndwdz.csv"
    #    tab = Table.read(filename,format="ascii.csv")
    #    rews = tab['rew']




#
    #if spline_interp is None:
    #    filename = resource_filename('MgIIabs.model','')[:-5]+"data/gauss_params.csv"
    #    params = Table.read(filename,format="ascii.csv")
    #    x = params['z']
    #    y = array([params['A'],params['mu'],params['sigma']])
    #    spline_interp = interp1d(x,y,kind="cubic")
    #A, mu, sigma = spline_interp(z)
#
    #M_low1 = log10(lowest_mass(rew_min,low=3,z=z).value/1e12)
    #f = lambda x: ndtr(-x)
#
    #if isinf(rew_max):
    #    integral = A*_dldz(z)*quad(f,(M_low1+mu)/sigma,float('inf'))[0]
    #else:
    #    M_low2 = log10(lowest_mass(rew_max,low=0,z=z).value/1e12)
    #    integral = quad(f,(M_low1+mu)/sigma,(M_low2+mu)/sigma)[0]*A*_dldz(z).value
    #return integral       
#
#def# d2ndWdl(rew,z=0.0,growthf=None,spline_interp=None):
##    """
#    This evaluates the integral for $d^2N/dW_rdl$
#    by approximating the integrand in log-normal
#    distribution. The integrand in this approximation
#    turns out to be the errorfunction simply.
#    Parameters
#    ----------
#    rew: astropy.Quantity
#        Rest equivalent width in nanometers
#    z: float, optional
#        Redshift
#    growthf: float, optional
#        The growthfactor corresponding to 
#        the input redshift. The function
#        works faster if supplied.
#    spline_interp: function
#        The interpolating function for the
#        gaussian parameters. If not given,
#        it will be computed. Works faster
#        if supplied.
#    Returns
#    -------
#    integral: astropy.Quantity
#        The integral in units of
#        $nm^{-1} h Mpc^{-1}$
#    """
#    from compos import growthfactor as gf
#    from astropy.table import Table
#    from numpy import log10, array
#    from pkg_resources import resource_filename
#    from scipy.interpolate import interp1d
#    from scipy.special import ndtr
#    from .halomodel import lowest_mass
#
#    #A bit costly. Best to provide
#    #these things to speed up calculations
#    if growthf is None:
#        growthf = gf.growfunc_z(z)
#    if spline_interp is None:
#        filename = resource_filename('MgIIabs.model','')[:-5]+"data/gauss_params.csv"
#        params = Table.read(filename,format="ascii.csv")
#        x = params['z']
#        y = array([params['A'],params['mu'],params['sigma']])
#        spline_interp = interp1d(x,y,kind="cubic")
#    A, mu, sigma = spline_interp(z)
#
#    M_low = log10(lowest_mass(rew,low=3,z=z).value/1e12)
#
#    integral = A*ndtr(-(M_low+mu)/sigma)
#    return integral/u.nm/u.Mpc
#

def opt_masses(z):
    from scipy.optimize import curve_fit
    from .observe import obs_d2ndwdz as zhu_men
    import pdb
    x = linspace(0.1,0.3,10)
    y = log10(zhu_men(x*10,z))

    def vec_d2(rewlist,z,kappa_m):
        d2list = asarray([d2ndWdz(rew*u.nm,z,kappa_m=kappa_m).value/10 for rew in rewlist])
        return d2list
    func = lambda rew,k1,k2,k3,k4: log10(vec_d2(rew,z,[k1,k2,k3,k4]))
    opt = curve_fit(func,x,y,p0=[9,10,11,12],bounds=(8,14))
    return opt