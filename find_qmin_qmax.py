import numpy as np
from get_header_footer import get_header_footer as ghf

def find_qmin_qmax(path,data,extension):
    qmin_data = 99
    qmax_data = 0
    for datafile in data:
        filename = '%s%s%s' % (path,datafile,extension)
        header,footer = ghf(filename)
        q,I,dI = np.genfromtxt(filename,skip_header=header,skip_footer=footer,unpack=True)
        if np.amin(q) < qmin_data:
            qmin_data = np.amin(q)
        if np.amax(q) > qmax_data:
            qmax_data = np.amax(q)
    return qmin_data,qmax_data

 