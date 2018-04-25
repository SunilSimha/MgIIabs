import numpy as np
from scipy.integrate import quad, simps, quadrature, romb, romberg
from scipy.optimize import curve_fit
from scipy import interpolate as interp
from compos import const, matterps as mps, growthfactor as gf
from astropy.units.astrophys import Mpc, M_sun
import astropy.units as u
from pkg_resources import resource_filename
from astropy.constants import G as grav
from astropy.table import Table
from MgIIabs.model import halomassfunc as hmf, halomodel as hmod, absdistrib as ad
import matplotlib.pyplot as plt
import pdb
from scipy.special import ndtr

import cProfile
import time

const.initializecosmo()
H0 = 100*u.km/u.s/u.Mpc #(units of h km/s/Mpc)
rho_cr0 = 3*H0**2/(8*np.pi*grav)
rho_m0 = const.cosmo['omega_0']*rho_cr0
rewlist = np.linspace(0.1,0.25,10)*u.nm
redshifts = np.linspace(0,2.5)
plt.figure()
all_integs = []

def integrand(logM_12):
    M = 10**logM_12*1e12
    dndm = hmf.dNdM(M*M_sun,z=z,growthf=growthf)*rho_m/M**2#Because of the way dN/dm is coded
    sigmag = np.pi*hmod.rg(M*M_sun,z=z).value**2
    pWM = hmod.p_rew_given_m(rew,M*M_sun,z=z)[0].value
    return dndm*sigmag*pWM*M

for z in redshifts:
    rho_m = rho_m0.to(M_sun/Mpc**3).value*(1+z)**3
    growthf = gf.growfunc_z(z)
    integlist = []
    dldz = ad._dldz(z).value
    for rew in rewlist:
        integlist.append(np.log(10)*quad(integrand,-2,2,points=[-0.67,0.66])[0]*dldz)
    all_integs.append(integlist)
    plt.semilogy(rewlist,integlist,label="{:1.1f}".format(z))
plt.legend(title="Redshifts")
plt.xlabel("REW (nm)")
plt.ylabel(r"$d^2N/dW_rdz$ ($nm^{-1}$)")
plt.show()

dndzdata = np.asarray(all_integs).transpose()
names = [str(z) for z in redshifts]
table = Table(dndzdata,names=names)
table['rew']=rewlist
filename = resource_filename('MgIIabs.count',"")[:-5]+'/data/d2ndwdz.csv'
table.write(filename,format="ascii.csv",overwrite=True)