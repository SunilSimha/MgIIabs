import numpy as np
from astropy.table import Table
from compos import growthfactor as gf
from pkg_resources import resource_filename
import os
import pdb

datafile = resource_filename('MgIIabs','data/growthfactor.ascii')

z = np.linspace(0,2.5,1000000+1)
gflist = []
for red in z:
    gflist.append(gf.growfunc_z(red))

t = Table()
t['z'] = z
t['gf'] = gflist

t.write(datafile,format='ascii')
os.system('spd-say "I\'m done writing the grwthfactors to disk"')