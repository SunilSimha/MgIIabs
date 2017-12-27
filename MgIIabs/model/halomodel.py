"""
Modelling the halo distribution of clouds. See Tinker and Chen ApJ 2008
"""

def rew_of_s(s,A_w,G0,Rg,ah):
    """
    Defines the relationship between the REW
    of MgII 2796 and the impact parameter assuming
    an isothermal profile.
    Parameters
    ----------
    s : float
        Impact parameter
    A_w : float
        Defined in eq. 6, Tinker & Chen 2008, ApJ
    G0 : float
        Normalization factor for halo density.
    Rg : float
        Effective gas alo radius
    ah : float
        Halo core radius
    Returns
    -------
    rew : float
        Rest equivlent width of absorption for
        the given impact parameter and halo parameters.
    """
    import numpy as np
    assert s>=0, "Impact parameter cannot be negative"
    if s>Rg:
        return 0
    else:
        return A_w*2*G0/np.sqrt(s**2+ah**2)*np.arctan(np.sqrt((Rg**2-s**2)/(s**2+ah**2)))

def s_of_rew(rew,A_w,G0,Rg,ah):
    """
    Inverse function of rew_of_s. Uses brentq for root finding.
    Parameters
    ----------
    rew : float
        rest equivalent width of absorption
    A_w : float
        Defined in eq. 6, Tinker & Chen 2008, ApJ
    G0 : float
        Normalization factor for halo density.
    Rg : float
        Effective gas alo radius
    ah : float
        Halo core radius
    Returns
    -------
    s : float
        Impact parameter
    """
    import numpy as np
    from scipy.optimize import brentq
    g = lambda s: rew_of_s(s,A_w,G0,Rg,ah)-rew
    try:
        return brentq(g,0,Rg)
    except(ValueError):
        print("s={:f} cannot be achieved with this model".format(rew))

def kappa_g(M):
    """
    This total probability density of finding a halo
    of mass M. For the classical model, this is a c-spline
    interpolation of between four different masses
    logM1 = 10.0, logM2 = 11.33, logM3 = 12.66, logM4 = 14.0
    (in solar masses) in log-log space
    Parameters
    ----------
    M : float or numpy array
        Halo mass
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

    output_points = 10**interpolate.splev(M,spline_interp)
    return output_points

def _ds_drew(s,ah,Rg,G0,Aw):
    """
    Returns ds/d(rew)
    """
    import numpy as np
    x = lambda s: np.sqrt((Rg**2-s**2)/(ah**2+s**2))
    y = lambda s: np.sqrt((ah**2+s**2))
    dsdrew = 1/(2*Aw*G0)/(s*np.arctan(x(s))/y(s)**3 + (1+x(s)**2)*y(s)*s*x(s)/(Rg**2-s**2)/y(Rg)**2)
    return dsdrew

def p_rew_given_m(rew,M,ah,Rg,G0,Aw):
    """
    Returns the conditional probability P(REW|M) for
    the given value of M and REW.

    Parameters
    ----------
    rew : float
        rest equivalent width of absorption
    M : float
        halo mass
    Aw : float
        Defined in eq. 6, Tinker & Chen 2008, ApJ
    G0 : float
        Normalization factor for halo density.
    Rg : float
        Effective gas alo radius
    ah : float
        Halo core radius
    Returns
    -------
    p : float
        P(REW|M) = kappa_g(M)*2s(REW|M)/Rg^2*ds/d(REW)
    """
    import numpy as np
    s = s_of_rew(rew,Aw,G0,Rg,ah)
    k = kappa_g(M)
    dsdrew = _ds_drew(s,ah,Rg,G0,Aw)
    p = k*2*s/Rg**2*dsdrew
    return p
