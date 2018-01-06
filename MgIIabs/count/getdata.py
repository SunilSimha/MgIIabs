"""
A module for the retreival and pre-processing of LoS data
obtained from Ravi and the MC Simulation data from G. B. 
Zhu.
"""

def relVec (z1, z2):
    """
    Velocity difference between two redshifts
    assuming Hubble expansion in units of c
    Parameters
    ----------
    z1, z2: float
        The two redshifts
    Returns
    -------
    beta: float
        Relative radial velocity in units of c
    """
    beta = ((z1+1)**2-(z2+1)**2)/((z1+1)**2+(z2+1)**2)
    return beta

def zRel (z0, betaRel):
    """
    An inversion of the `reVec` function.
    Parameters
    ---------
    z0: float
        A redshift
    betaRel: float
        Relative radial velocity
    Returns
    -------
    z1: float
        Redshift
    """
    z1 = ((1+z0)*np.sqrt((1+betaRel)/(1-betaRel))-1)
    return z1

def getObsData(zmin=0,zmax=float('inf')):
    """
    This function obtains the data such that the absorbers lie 
    in the redshift range [zmin,zmax].
    Parameters
    ----------
    zmin, zmax: float, optional
        Lower and upper limits of redshift to retreive data
        from.
    Returns
    -------
    fullData: astropy.fits table
    """
    import os
    from astropy.io import fits
    try: 
        data_folder = os.environ['MGIIDATA']
    except TypeError:
        print("MGIIDATA env variable not set")

    hdulist = fits.open(data_folder+'Trimmed_SDSS_DR7_107_NonBAL_NonRpt_SDSSRpt.fits')
    #Importing the binary table from the fits file opened
    fullData = hdulist[1].data
    #Imposing the non-BAL condition. I'll use 'bal_flg' field in the
    #data for this.
    fullData = fullData[fullData['bal_flg']==0]
    #Imposing redshift range limits
    fullData = fullData[(fullData['ZABS']>=zmin)*(fullData['ZABS']<=zmax)]
    #Filtering out associated absorbers (<5000 km/s away)
    fullData = fullData[abs(relVec(fullData['ZABS'],
                        fullData['ZQSO']))>(5000/3e5)]
    return fullData

def getZgrid(zmin = 0, zmax = float('inf')):
    """
    The theoretical calculations are done on the same grid used by Zhu
    in the MC simulations.
    Parameters
    ----------
    zmin, zmax: float, optional
        Lower and upper limits of redshifts.
        Default values: 0, inf
    Returns
    -------
    zgrid: numpy array
        Grid of redshfits.
    """
    import os
    from astropy.io import fits
    try: 
        data_folder = os.environ['MGIIDATA']
    except TypeError:
        print("MGIIDATA env variable not set")

    a = fits.open(data_folder+"MC data from G Ben Zhu/All_in_One_QSO_spec_MonteCarlo_wmin_106_zgrid.fits")
    zgrid = a[1].data['ZGRID'][0]
    zgrid = zgrid[zgrid >= zmin]
    zgrid = zgrid[zgrid <= zmax]
    return zgrid

def preProcess(minREW,fullData,zmin=0,zmax=float('inf')):
    """
    This function takes input as the minimum REW
    to serve as a threshold and the width of 
    the histogram bins and return the observed
    and theoretical histograms (uniform bin sizes).
    Parameters
    ----------
    minREW: float
        lower limt or Mg II 2796 REW.
    fullData: astropy.io.fit record
        Output it or Mg II 2796 REW.
    zmin: float, optional
        The minimum value of absorber redshift to be considered.
        Default is 0.
    zmax: float, optinal
        The maximum value of absorber redshift to be considered.
         Default is float('inf')
    Returns
    -------
    final_list: list
        A list of LoS with the following data
        1) A list of absorbers with redshifts in the specified range.
        2) A grid of covered redshifts (from the MC data)
        3) A grid of minimum observable REW at each redshift in the range.
    """
    import os
    from astropy.io import fits
    from astropy.table import Table
    #--------------------------------------------------------------------------
    #First to get the MC data from Zhu et al's simulations.
    #This shall be stored in fullMC
    rootDir = "MC data from G Ben Zhu/All_in_One_QSO_spec_MonteCarlo_wmin_106_0"
    
    temp = fits.open(rootDir+str(1)+".fits",memmap=True); #File number 1
    fullMC = sorted(temp[1].data, key = itemgetter(0));
    
    for i in np.arange(2,9) :#There are 9 files, hence 2-9
        temp = fits.open(rootDir+str(i)+".fits");
        temp2 = sorted(temp[1].data, key = itemgetter(0));
        fullMC = fullMC + temp2;
    del temp, temp2
    
    #--------------------------------------------------------------------------
    #Now to get the absorber data above the required minimum REW limit.
    truncData = fullData[fullData['REW_MGII_2796'] > minREW]
    #-------------------------------------------------------
    #Now to get the unique values of PLATE, FIBER, MJD
    #First to get all the values
    uniquePFMJD = np.array([truncData['PLATE'],
                            truncData['FIBER'],
                            truncData['MJD']]).T
    
    #Next to get the unique values. This pythonesque method
    #is credited to an answer found on stackexchange
    uniquePFMJD = np.vstack({tuple([row[0],row[1],row[2]]) 
                             for row in uniquePFMJD})
    #-------------------------------------------------------
    #This is the list of PLATE, FIBER values in the complete
    #MC simulation 
    MC_PFlist = np.array([[a[0],a[1]] for a in fullMC])
    #-------------------------------------------------------
    
    final_list = [] #initialize an empty list
    
    #Now to get the range of z values.
    zgrid = getZgrid()
    zrange = (zgrid>=zmin)*(zgrid<=zmax)#A boolean array
    del zgrid
    
    for row in uniquePFMJD:
        #This obtains the indices of truncData which correspond
        #to the same PLATE, FIBER, MJD values as those of uniquePFMJD's row
        temp_data_indices = np.arange(len(truncData))[(truncData['PLATE']==row[0])
                                                      *(truncData['FIBER']==row[1])
                                                      *(truncData['MJD']==row[2])]
        #-------------------------------------------------------
        #This returns indices from fullMC that correspond to
        #those PLATE, FIBER values: 
        temp_MC_indices = np.arange(len(fullMC))[(MC_PFlist[:,0]==row[0])
                                                 *(MC_PFlist[:,1]==row[1])]
        #-------------------------------------------------------
        #It could be that this particular pair of values may not
        #exist in the simulated data. So, in case it exists:
        if (temp_MC_indices.tolist()!=[]):
            #Get the corresponding absorbers' redshift from truncData
            temp_zabs = itemgetter(*temp_data_indices)(truncData['ZABS'])
            #------------------------------------------------------
            #Dump the absorber redshift list, the covered fraction
            #and the minimum REW visible at the redshift grid into
            #final_list:
            coverage = fullMC[temp_MC_indices[0]][2][zrange]
            minREWgrid = fullMC[temp_MC_indices[0]][3][zrange]
            final_list = final_list + [[temp_zabs,
                                        coverage,
                                        minREWgrid]]
            #----------------------------------------------------------
    return final_list