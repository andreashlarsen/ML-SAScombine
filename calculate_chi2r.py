import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit

def calculate_p_value(reduced_chi_square,dof):
    p_value = stats.chi2.sf(reduced_chi_square*dof, dof)
    return p_value

def calculate_chi2r(R,dof):
    chi2 = np.sum(R**2)
    chi2r = chi2/dof
    p = calculate_p_value(chi2r,dof)
    return chi2r,p

def trunc(q,I,dI,q_ref):
    idx = np.where((q <= np.amax(q_ref)) & (q >= np.amin(q_ref)))
    q_t,I_t,dI_t = q[idx],I[idx],dI[idx]
    return q_t,I_t,dI_t

def fit_scale_offset(q_t,I_t,dI_t,q_ref,I_ref):

    ## interpolate ref data on q-values
    I_interp = np.interp(q_t,q_ref,I_ref)
    
    ## function for fitting
    def lin_func(q_t,a,b):
        return a*I_interp+b

    ## initial guesses 
    a0,b0 = I_t[0]/I_interp[0],I_t[-1]-I_interp[-1]
    
    ## fitting
    popt,pcov = curve_fit(lin_func,q_t,I_t,sigma=dI_t,p0=[a0,b0])
    I_interp_fit = lin_func(q_t,*popt)
    
    return I_interp_fit,popt
