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
