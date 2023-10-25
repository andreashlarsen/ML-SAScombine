import scipy.stats as stats
import numpy as np

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
