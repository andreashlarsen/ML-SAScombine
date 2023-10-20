import numpy as np

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