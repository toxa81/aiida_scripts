import numpy as np

#=======================
# Birch-Murnaghan EOS
#=======================
def birch_murnaghan(V,E0,V0,B0,B01):
    r = (V0/V)**(2./3.)
    return (E0 + 9./16. * B0 * V0 * (r-1.)**2 * ( 2.+ (B01-4.)*(r-1.)))

#=======================
# function to fit the EOS
#=======================
def fit_birch_murnaghan_params(volumes, energies):
    x = np.array(volumes)
    y = np.array(energies)
    from scipy.optimize import curve_fit

    b01 = 0.1
    b01min = 0
    perrmin = 100

    #while True:
    #    params, covariance = curve_fit(birch_murnaghan,
    #                                   xdata=x,
    #                                   ydata=y,
    #                                   p0=(y.min(),    #E0
    #                                       x.mean(),   #V0
    #                                       0.1,        #B0
    #                                       b01,         #B01
    #                                   ),
    #                                   sigma=None)
    #    
    #    if isinstance(covariance, np.ndarray):
    #        perr = np.sqrt(np.diag(covariance))
    #        if np.sum(perr) < perrmin: 
    #            perrmin = np.sum(perr)
    #            b01min = b01
    #        #print("b01, perr: ", b01, np.sum(perr))
    #        #if perr[1] < 0.1: break


    #    b01 += 0.1
    #    if b01 > 10: break
    
    try:
        params, covariance = curve_fit(birch_murnaghan, xdata=x, ydata=y,
                                       p0=(y.min(), x.mean(), 0.0, 0.0),
                                       sigma=None)
        return params
    except:
        print("fit_birch_murnaghan_params: failed")
        return None

def deltaE2(V, params1, params2):
    return (birch_murnaghan(V, *params1) - birch_murnaghan(V, *params2))**2
