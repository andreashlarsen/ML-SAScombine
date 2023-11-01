import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
import os

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

## smooth funciton
def smooth(x,n,type):

    # ensure n is odd
    if (n % 2) == 0:
        n += 1

    # calculate wing size
    wing_size = int((n-1)/2)

    # initiate x_smooth (no need to change first and last value)
    x_smooth = x.copy() # brackets ensures list x is not edited
    j = 1
    rest = len(x)-j
    while j < len(x)-1:
        if (j < wing_size) or (rest-1 < wing_size) > j:
            i_max = j
        elif rest-1 < wing_size:
            i_max = rest-1
        else:
            i_max = wing_size
        
        if type == 'lin':
            dw = 1/(wing_size+1)
        
        sum_w = 1
        sum = x[j]
        for i in range(1,i_max+1):
            if type == 'lin':
                w = 1 - i*dw
            elif type == 'uni':
                w = 1
            else:
                print('ERROR in function smooth(): type (3rd argument, string value) must be lin for linear or uni for uniform')
                exit()
            sum_w += 2*w
            #sum += x[j-i] + x[j+i]
            sum += w*(x[j-i] + x[j+i])
        #x_smooth[j] = sum/(2*i_max+1)
        x_smooth[j] = sum/sum_w
        j += 1
        rest = len(x) - j

    return x_smooth

def get_header_footer(file):
    """
    get number of headerlines and footerlines
    """

    header,footer = 0,0
    f = open(file)
    try:
        lines = f.readlines()
    except:
        print('Error: cannot read lines of file. Do you have some special characters in the file? Try removing them and rerun')
        print('file: %s' % file)

    CONTINUE_H,CONTINUE_F = True,True
    j = 0
    while CONTINUE_H or CONTINUE_F:
        line_h = lines[j]
        #print(line_h)
        line_f = lines[-1-j]
        tmp_h = line_h.split()
        tmp_f = line_f.split()
        try:
            NAN = 0
            for i in range(len(tmp_h)):
                1/float(tmp_h[i]) # divide to ensure non-zero values
                if np.isnan(float(tmp_h[i])):
                    NAN = 1
            if not tmp_h:
                NAN = 1 #empty line

            if NAN:
                header+=1
            else:
                CONTINUE_H = False
        except:
            header+=1
        try:
            NAN = 0
            for i in range(len(tmp_f)):
                1/float(tmp_f[i]) # divide to ensure non-zero values
                if np.isnan(float(tmp_f[i])):
                    NAN = 1
            if not tmp_h:
                NAN = 1 #empty line
                
            if NAN:
                footer+=1
            else:   
                CONTINUE_F = False
        except:
            footer+=1
        j+=1

    return header,footer

def find_qmin_qmax(path,data,extension,RANGE):
    """
    find minimum and maximum q of a set of data
    if RANGE, then find second smallest/largest qmin/qmax
    """
    qmin_list,qmax_list = [],[]
    for datafile in data:
        filename = '%s%s%s' % (path,datafile,extension)
        if not os.path.exists(filename):
            filename = '%s/%s%s' % (path,datafile,extension)
            if not os.path.exists(filename):
                filename = '%s%s.%s' % (path,datafile,extension)
                if not os.path.exists(filename):
                    filename = '%s/%s.%s' % (path,datafile,extension)
                    if not os.path.exists(filename):
                        filename = '%s%s%s' % (path,datafile,extension)
        header,footer = get_header_footer(filename)
        q,I,dI = np.genfromtxt(filename,skip_header=header,skip_footer=footer,unpack=True)
        qmin_list.append(np.amin(q))
        qmax_list.append(np.amax(q))
    if RANGE:
        qmin_list.sort()
        qmax_list.sort()
        qmin = qmin_list[1]
        qmax = qmax_list[-2]
    else:
        qmin = np.amin(qmin_list)
        qmax = np.amax(qmax_list)

    return qmin,qmax

def add_data(q_sum,I_sum,w_sum,q,I_fit,dI_fit,q_edges):
    M = len(q)
    j = 0    
    w = dI_fit**-2
    for j in range(M):
        try:
            idx = np.where(q_edges<q[j])[0][-1]
        except:
            idx = 0
        q_sum[idx] += w[j]*q[j]
        I_sum[idx] += w[j]*I_fit[j]
        w_sum[idx] += w[j] 

def append_data(q_matrix,I_matrix,dI_matrix,w_matrix,q,I_fit,dI_fit,q_edges):
    M = len(q)
    j = 0    
    w = dI_fit**-2
    for j in range(M):
        try:
            idx = np.where(q_edges<q[j])[0][-1]
        except:
            idx = 0
        q_matrix[idx].append(q[j])
        I_matrix[idx].append(I_fit[j])
        dI_matrix[idx].append(dI_fit[j])
        w_matrix[idx].append(w[j]) 