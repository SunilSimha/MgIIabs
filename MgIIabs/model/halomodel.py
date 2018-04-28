"""
Modelling the halo distribution of clouds. See Tinker and Chen ApJ 2008
"""
#Include astropy units
import astropy.units as u
from astropy.constants import G as grav
from astropy.units.astrophys import Mpc, Msun
from astropy.cosmology import Planck13
from numpy import pi, arctan, sqrt, asarray, log10
from halotools.empirical_models import NFWProfile
import pdb

h = Planck13.h
H0 = 100*u.km/u.s/u.Mpc #(units of h km/s/Mpc)
rho_cr0 = 3*H0**2/(8*pi*grav)
rho_m0 = Planck13.Om0*rho_cr0

def rg(M,z=0):
    """
    Effective halo gas radius. A simple scaling relation
    with mass.
    Parameters
    ----------
    M: astropy.Quantity
        Halo mass in units of h^-1 Msun
    z: float, optional
        Redshift
    Returns
    -------
    r: astropy.Quantity
        Effective gas radius in Mpc/h 
    """
    #Because the comoving radius of the halo is redshift independent,
    return 0.08/(1+z)*Mpc*(M/1e12/Msun)**(1/3) #*(1+z)**-0.2

def Aw(M, A_w0=9*u.nm*(u.cm)**2/u.g,z=0):
    """
    Fudge factor (for the classical model)
    Parameters
    ----------
    M: astropy.Quantity
        Halo mass
    A_w0: astropy.Quantity
        13.0 h nm cm^2/g by default
    Returns
    -------
    A_w: astropy.Quantity
        In units of h nm cm^2/g
    """
    if M<=1e12*Msun:
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
    #Initializing NFW everytime might be
    #slowing it down. Consider initializing
    #outside and updating z.
    # 
    # Cannot be done natively. :(
    nfw = NFWProfile(cosmology=Planck13,redshit=z,mdef='200m',conc_mass_model='dutton_maccio14')
    conc = nfw.conc_NFWmodel(prim_haloprop=M.to(Msun).value/h,mdef='200m')

    Rg = rg(M,z)
    m_enc = nfw.enclosed_mass(Rg.value/h,M.value/h,conc)*Msun
    ah = ah_by_Rg*Rg

    G0 = m_enc/(4*pi)/(Rg-ah*arctan(1/ah_by_Rg))
    return G0.to(Msun/Mpc)

def rew_of_s(s,M,ah_by_Rg=0.2,A_w0=9*u.nm*(u.cm)**2/u.g,z=0):
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
    assert(s.value>=0), "Impact parameter cannot be negative"

    Rg = rg(M,z)
    A_w = Aw(M,A_w0,z)
    ah = ah_by_Rg*Rg
    G0 = g0(M,z,ah_by_Rg)

    if s.to(Mpc).value>=Rg.value:
        return 0*u.nm
    else:
        rew = A_w*2*G0/sqrt(s**2+ah**2)*arctan(sqrt((Rg**2-s**2)/(s**2+ah**2))).value
        return rew.to(u.nm)

def lowest_mass(rew,low=0,high=16,ah_by_Rg=0.2,A_w0=9*u.nm*(u.cm)**2/u.g,z=0):
    """
    Finds the lowest halo mass for which
    the input rest equivalent width is possible
    Parameters
    ----------
    rew: astropy.Quantity
        Rest eauivalent width (nm)
    low, high: float,optional
        lower and upper logarithmic_10 limits
        of search window of mass (h^-1 Msun)
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
        Halo mass (Msun)
    """
    from scipy.optimize import brentq

    f = lambda logM: rew_of_s(0*Mpc,10**logM*Msun,ah_by_Rg,A_w0,z).value - rew.value
    try:
        return 10**brentq(f,low,high)*Msun
    except(ValueError):
        raise ValueError("Cannot find a solution in the search window.")

def s_of_rew(rew,M,ah_by_Rg=0.2,A_w0=9*u.nm*(u.cm)**2/u.g,z=0):
    """
    Inverse function of rew_of_s. Uses brentq for root finding.
    Parameters
    ----------
    rew : astropy.Quantity
        rest equivalent width of absorption.
        (nanometers).
    M : astropy.Quantity
        Halo mass (Msun).
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
    from scipy.optimize import brentq

    g = lambda s: rew_of_s(s*Mpc,M,ah_by_Rg,A_w0,z).value-rew.value
    Rg = rg(M,z)
    try:
        # Really costly. Store results to speed up in case you
        # want to use this multiple times.
        return brentq(g,0,Rg.value)*Mpc
    except(ValueError):
        raise ValueError("rew={:f} cannot be achieved with this model".format(rew))

def kappa_g(M,kappa_m=None):
    """
    This total probability density of finding a halo
    of mass M. For the classical model, this is a c-spline
    interpolation of between four different masses
    logM1 = 10.0, logM2 = 11.33, logM3 = 12.66, logM4 = 14.0
    (in solar masses) in log-log space
    Parameters
    ----------
    M : astropy.Quantity
        Halo mass (Msun/h)
    Returns
    -------
    kappa : float or numpy array
        Probability density at that mass    
    """
    from scipy import interpolate
    #Data from
    kappa_array = asarray([-2,0,0,-2])
    if kappa_m is None:
        logm_array = asarray([10,11.33,12.66,14])
    else:
        logm_array = kappa_m   
    spline_interp = interpolate.interp1d(logm_array,kappa_array,bounds_error=False,kind='linear',fill_value="extrapolate")

    output_points = 10**spline_interp(log10(M.value))
    return output_points

def _ds_drew(s,M,ah_by_Rg=0.2,A_w0=9*u.nm*(u.cm)**2/u.g,z=0):
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

    Rg = rg(M,z)
    ah = ah_by_Rg*Rg
    G0 = g0(M,z,ah_by_Rg)
    A_w = Aw(M,A_w0,z)

    x = lambda s: sqrt((Rg**2-s**2)/(ah**2+s**2))
    y = lambda s: sqrt((ah**2+s**2))
    
    dsdrew = 1/(2*A_w*G0)/(s*arctan(x(s)).value/y(s)**3 + (1+x(s)**2)*y(s)*s*x(s)/(Rg**2-s**2)/y(Rg)**2)
    #pdb.set_trace()
    return dsdrew

def p_rew_given_m(rew,M,ah_by_Rg=0.2,A_w0=9*u.nm*(u.cm)**2/u.g,z=0,kappa_m=None):
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

    try:
        rew.to(u.nm)
    except:
        rew = rew*u.nm
    try:
        s = s_of_rew(rew,M,ah_by_Rg,A_w0,z)
    except(ValueError):
        p = 0/u.nm
        return p
    Rg = rg(M,z)
    k = kappa_g(M,kappa_m)
    dsdrew = _ds_drew(s,M,ah_by_Rg,A_w0,z)
    p = k*2*s/Rg**2*dsdrew  
    return p.to(1/u.nm)[0]
