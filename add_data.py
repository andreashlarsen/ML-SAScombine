import numpy as np

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