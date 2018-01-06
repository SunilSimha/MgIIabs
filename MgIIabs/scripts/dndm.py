import numpy as np
import matplotlib.pyplot as plt
from compos import const
from MgIIabs.model import halomassfunc
from astropy.units.astrophys import M_sun,Mpc
from scipy.optimize import minimize

M = 10**np.linspace(10,16,100)*M_sun #Units of h^-1 M_sun
plots = []
for z in np.arange(4):
    dndM = [halomassfunc.dNdM(mass,z) for mass in M]
    print('Done with z = {:d}'.format(z))
    plots.append(plt.loglog(M,dndM,label='{:d}'.format(z))[0])
plt.legend(handles=plots,title='Redshifts')
plt.ylabel(r'$\log (M^2/\rho_m dN/dM)$')
plt.xlabel(r'$\log(Mh/M_\odot)$')
plt.show()