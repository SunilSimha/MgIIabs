"""
This module contains the functions necessary to generate
the mass function using Press-Schechter formalism.
"""
#Note to self: In case of incorrectness of results, look
#at the constant factors.

#Need to test for redshift higher than zero.

def _test():
    """
    To test random things.
    """
    print("This is just to test stuff")



def window_th(k,R=1):
    """
    Normalised top-hat window function
    in the fourier domain. All lengths
    are in Mpc.
    Parameters
    ----------
    k: float
        Wavenumber
    R: float, optional
        Smoothing radius. By default
        1 Mpc
    Returns
    -------
    W: float
        Value of the window function
    """
    import numpy as np
    temp = k*R
    W = 3*(np.sin(temp)/(temp)**3-np.cos(temp)/(temp)**2)
    return W
    #Note to self: Try using astropy units


def psvariance(R, growthf=1, W='th',low = 1e-7,high=None,z=None,):
    """
    Variance of the linear power spectrum. Use the
    power spectrum from COMPOS (Ziang Yen). Numerically
    and obtains the variance.
    Parameters
    ----------
    R: float
        Smoothing radius
    W: string, optional
        Form of the normalised window function.
        'th' for top-hat profile and 'ga' for
        gaussian.
    low: float, optional
        Lower integration limit to get the variance
    high: float, optional
        Upper integration limit
    growthf: float
        Growthfactor for a non-zero redshift. If this is
        not provided, it is assumed to be at redshift 0.
    Returns
    -------
    variance: float
        Variance of the smoothed ps.
    error: float
        Error incurred in variance from numerical integration
    """
    import numpy as np
    from scipy.integrate import quad
    from compos import const, matterps, growthfactor
    import pdb
    if high is None:
        high = 20/R
    const.initializecosmo()
    #integrand = lambda k: (k*window_th(k,R))**2*gfactor*matterps.normalizedmp(k*const.cosmo['h'])
    integrand = lambda k: (k*window_th(k,R)*growthf)**2*matterps.normalizedmp(k*const.cosmo['h'])
    integral = quad(integrand,low,high)
    variance = integral[0]/(2*np.pi)**2
    error = integral[1]/(2*np.pi)**2
    return variance, error

#Include redshift dependence of the parameters.
def f_of_sigma(sigma,A=0.186,a=1.47,b=2.57,c=1.19,z=0,Delta=200):
    """
    The prefactor in the mass function parametrized
    as in Tinker et al 2008. The default values
    of the optional parameters correspond to a mean
    halo density of 200 times the background. The
    values can be found in table 2 of 
    Tinker 2008 ApJ 679, 1218
    Parameters
    ----------
    sigma: float
        Standard deviation of the linear power spectrum
    A,a,b,c: float, optional
        For Delta=200.
    z: float, optional 
        Redshift
    Delta: float, optional
        Halo mean overdensity
    Returns
    -------
    f: float
        Value of f(sigma) 

    """
    import numpy as np
    A = A*(1+z)**(-0.14)
    a = a*(1+z)**(-0.06)
    alpha = np.exp(-(0.075/np.log(Delta/75))**1.2)
    b = b*(1+z)**(-alpha)
    f = A*((sigma/b)**(-a)+1)*np.exp(-c/sigma/sigma)
    return f

def dlogsigma_dr(R,eps=None,**kwargs):
    """
    Numerical derivative of the log of standard
    deviation of the power spectrum with the smoothing
    radius. Uses central difference (accurate to the
    second order in eps) method.
    Parameters
    ----------
    R: float
        Smoothing radius
    eps: float, optional
        Spacing between the points used in the finite
        difference scheme. Default value is R*1e-8.
    **kwargs: optional
        Additional optional parameters for `psvariance`
    Returns
    -------
    dlnsdR: float
        Value of the derivative
    """
    import numpy as np
    if eps is None:
        eps = R*1e-8
    
    lns = lambda x: 0.5*np.log(psvariance(x,**kwargs)[0])
    dlnsdR = 0.5*(lns(R+eps)-lns(R-eps))/eps
    return dlnsdR

def dNdM(M,z=0,growthf=None,window='th'):
    """
    The halo mass function as given equation 2 of
    Tinker 2008 ApJ 679, 1218
    Parameters
    ----------
    M: float
        Halo mass in units of M_sun/h
    z: float, optional
        Redshift at which the mass funciton is to
        be computed. Default value: 0
    growthf: float, optional
        Provide the growthfactor at redshift z.
        If not provided, `dNdM` will calculate it but
        it will be slow if `dNdM` is being called
        multiple times.
    window: string, optional
        Choice of window function. By default it
        is a top hat profile.
    Returns
    -------
    dNdM: float
        dN/dM. Implicitly dependent of redshift.
        This is in units of M^2/rho_m. To get the
        true dN/dM, multiply by rho_m/M^2
    """
    import numpy as np
    from compos import const, growthfactor
    from astropy.units import Mpc, M_sun
    from astropy.constants import G as grav
    import astropy.units as u

    H0 = 100*u.km/u.s/Mpc
    rho_crit0 = (3*H0**2/(8*np.pi*grav)).to(M_sun/Mpc**3)

    const.initializecosmo()
    #rho_crit0 = 2.776992e12 #M_sun/Mpc^3
    rho_m = const.cosmo['omega_0']*rho_crit0*(1+z)**3
    R = (3*M/(4*np.pi*rho_m))**(1/3)
    R = R.to(Mpc).value
    if growthf is None:
        growthf = growthfactor.growfunc_z(z) 
    sigma = np.sqrt(psvariance(R,growthf)[0])

    dNdM = -f_of_sigma(sigma,z=z)*R*dlogsigma_dr(R,growthf=growthf)/3
    return dNdM