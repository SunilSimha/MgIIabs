"""
Modelling the halo distribution of clouds. See Tinker and Chen ApJ 2008
"""
#Include astropy units
import astropy.units as u
from astropy.constants import G as grav
from astropy.units.astrophys import Mpc, M_sun
from numpy import pi
from compos import const

const.initializecosmo()
H0 = 100*u.km/u.s/u.Mpc #(units of h km/s/Mpc)
rho_cr0 = 3*H0**2/(8*pi*grav)
rho_m0 = const.cosmo['omega_0']*rho_cr0

def rg(M,z=0):
    """
    Effective halo gas radius. A simple scaling relation
    with mass.
    Parameters
    ----------
    M: astropy.Quantity
        Halo mass
    z: float, optional
        Redshift
    Returns
    -------
    r: astropy.Quantity
        Effective gas radius in Mpc/h 
    """
    #Because the comoving radius of the halo is redshift independent,
    return 0.08*Mpc/(1+z)*(M/1e12/M_sun)**(1/3)

def Aw(M, A_w0=13*u.nm*(u.cm)**2/u.g):
    """
    Fudge factor (for the classical model)
    Parameters
    ----------
    M: astropy.Quantity
        Halo mass
    A_w0: astropy.Quantity
        13 h nm cm^2/g by default
    Returns
    -------
    A_w: astropy.Quantity
        In units of h nm cm^2/g
    """
    if M.value<=1e12:
        A_w = A_w0*(M.value/1e12)**(-0.172)
    else:
        A_w = A_w0*(M.value/1e12)**(-0.176)
    return A_w


def g0(M,z=0,ah_by_Rg=0.2):
    """
    Gas mass normalization constant
    Parameters
    ----------
    M: astropy.Quantity
        Halo mass
    z: float, optional
        Redshift. Default: 0
    ah_by_rg: float, optional
        Ratio of core radius to
        effective gas radius
    Returns
    -------
    G0: astropy.Quantity
        Normalization constant for
        an isothermal halo profile
        of gas radius rg(M) and core
        radius ah (Msun/Mpc^2)
    """
    from numpy import pi, arctan
    from astropy.cosmology import Planck13
    from halotools.empirical_models import NFWProfile

    nfw = NFWProfile(cosmology=Planck13,redshit=z,mdef='200m',conc_mass_model='dutton_maccio14')
    conc = nfw.conc_NFWmodel(prim_haloprop=M.value)

    Rg = rg(M,z)
    m_enc = nfw.enclosed_mass(Rg.value,M.value,conc)*M_sun
    ah = ah_by_Rg*Rg

    G0 = m_enc/(4*pi)/(Rg-ah*arctan(1/ah_by_Rg))
    return G0.to(M_sun/Mpc)

def rew_of_s(s,M,ah_by_Rg=0.2,A_w0=13*u.nm*(u.cm)**2/u.g,z=0):
    """
    Defines the relationship between the REW
    of MgII 2796 and the impact parameter assuming
    an isothermal profile.
    Parameters
    ----------
    s : astropy.Quantity
        Impact parameter in Mpc
    M : astropy.Quantity
        Halo mass.
    ah_by_Rg : float, optional
        Ratio of core radius to effective
        gas radius
    A_w0 : float, optional
        Defined in eq. 6, Tinker & Chen 2008, ApJ
        In units of  h nm cm^2/g
    z: float, optional
        Redshift
    Returns
    -------
    rew : float
        Rest equivlent width of absorption for
        the given impact parameter and halo parameters.
    """
    import numpy as np
    assert(s.value>=0), "Impact parameter cannot be negative"

    Rg = rg(M)
    A_w = Aw(M,A_w0)
    ah = ah_by_Rg*Rg
    G0 = g0(M,z,ah_by_Rg)

    if s.value>Rg.value:
        return 0*u.nm
    else:
        rew = A_w*2*G0/np.sqrt(s**2+ah**2)*np.arctan(np.sqrt((Rg**2-s**2)/(s**2+ah**2))).value
        return rew.to(u.nm)

def lowest_mass(rew,low=8,high=16,ah_by_Rg=0.2,A_w0=13*u.nm*(u.cm)**2/u.g,z=0):
    """
    Finds the lowest halo mass for which
    the input rest equivalent width is possible
    Parameters
    ----------
    rew: astropy.Quantity
        Rest eauivalent width (nm)
    low, high: float,optional
        lower and upper logarithmic_10 limits
        of search window of mass 
    ah_by_Rg : float, optional
        Ratio of core radius to effective
        gas radius
    A_w0 : float, optional
        Defined in eq. 6, Tinker & Chen 2008, ApJ
        In units of  h nm cm^2/g
    z: float, optional
        Redshift
    Returns
    -------
    M: astropy.Quantity
        Halo mass (M_sun)
    """
    from scipy.optimize import brentq

    f = lambda logM: rew_of_s(0*Mpc,10**logM*M_sun,ah_by_Rg,A_w0,z).value - rew.value
    try:
        return 10**brentq(f,low,high)*M_sun
    except(ValueError):
        raise ValueError("Cannot find a solution in the search window.")

def s_of_rew(rew,M,ah_by_Rg=0.2,A_w0=13*u.nm*(u.cm)**2/u.g,z=0):
    """
    Inverse function of rew_of_s. Uses brentq for root finding.
    Parameters
    ----------
    rew : astropy.Quantity
        rest equivalent width of absorption.
        (nanometers).
    M : astropy.Quantity
        Halo mass (M_sun).
    ah_by_Rg: float, optional
        Ratio of core radius to effective gas
        radius
    A_w0 : astropy.Quantity, optional
        Defined in eq. 6, Tinker & Chen 2008, ApJ
        (h nm cm^2/g).
    z: float, optional
        Redshift
    Returns
    -------
    s : float
        Impact parameter (Mpc/h)
    """
    import numpy as np
    from scipy.optimize import brentq
    import pdb

    g = lambda s: rew_of_s(s*Mpc,M,ah_by_Rg,A_w0,z).value-rew.value
    Rg = rg(M,z)
    try:
        return brentq(g,0,Rg.value)*Mpc
    except(ValueError):
       raise ValueError("rew={:f} cannot be achieved with this model".format(rew))

def kappa_g(M):
    """
    This total probability density of finding a halo
    of mass M. For the classical model, this is a c-spline
    interpolation of between four different masses
    logM1 = 10.0, logM2 = 11.33, logM3 = 12.66, logM4 = 14.0
    (in solar masses) in log-log space
    Parameters
    ----------
    M : astropy.Quantity
        Halo mass (M_sun/h)
    Returns
    -------
    kappa : float or numpy array
        Probability density at that mass    
    """
    import numpy as np
    from scipy import interpolate
    #Data from
    m_array = np.asarray([10.0,11.33,12.66,14.0])
    kappa_array = np.asarray([-1.721,-0.012,-0.198,-1.763])
    spline_interp = interpolate.splrep(m_array,kappa_array)

    output_points = 10**interpolate.splev(np.log10(M.value),spline_interp)
    return output_points

def _ds_drew(s,M,ah_by_Rg=0.2,A_w0=13*u.nm*(u.cm)**2/u.g,z=0):
    """
    Returns ds/d(rew)
    Parameters
    ----------
    rew : astropy.Quantity
        rest equivalent width of absorption
    M : astropy.Quantity
        halo mass
    ah_by_Rg: float, optional
        Ratio of core radius to effective gas
        radius
    A_w0 : astropy.Quantity, optional
        Defined in eq. 6, Tinker & Chen 2008, ApJ
    z: float, optional
        Redshift
    Returns
    -------
    ds/d(rew): float
        Rate of change of impact parameter with
        rest equivalent width.
    """
    import numpy as np

    Rg = rg(M)
    ah = ah_by_Rg*Rg
    G0 = g0(M,z,ah_by_Rg)
    A_w = Aw(M,A_w0)
    x = lambda s: np.sqrt((Rg**2-s**2)/(ah**2+s**2))
    y = lambda s: np.sqrt((ah**2+s**2))
    dsdrew = 1/(2*A_w*G0)/(s*np.arctan(x(s)).value/y(s)**3 + (1+x(s)**2)*y(s)*s*x(s)/(Rg**2-s**2)/y(Rg)**2)
    return dsdrew

def p_rew_given_m(rew,M,ah_by_Rg=0.2,A_w0=13*u.nm*(u.cm)**2/u.g,z=0):
    """
    Returns the conditional probability P(REW|M) for
    the given value of M and REW.

    Parameters
    ----------
    rew : astropy.Quantity
        rest equivalent width of absorption
    M : astropy.Quantity
        halo mass
    ah_by_Rg: float, optional
        Ratio of core radius to effective gas
        radius
    A_w0 : astropy.Quantity, optional
        Defined in eq. 6, Tinker & Chen 2008, ApJ
    z: float, optional
        Redshift
    Returns
    -------
    p : float
        P(REW|M) = kappa_g(M)*2s(REW|M)/Rg^2*ds/d(REW)
    """
    import numpy as np
    s = s_of_rew(rew,M,ah_by_Rg,A_w0,z)
    Rg = rg(M,z)
    k = kappa_g(M)
    dsdrew = _ds_drew(s,M,ah_by_Rg,A_w0,z)
    p = k*2*s/Rg**2*dsdrew  
    return p.to(1/u.nm)
