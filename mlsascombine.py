#!/usr/bin/python3

#############################
# ML-SAScombine, version:
version = 'beta0.16'
#############################

## importing python packages
try: 
    import sys
except:
    print("ERROR: ML-SAScombine tried to import python package sys - is it correctly installed?\n")
    import sys
try:
    import argparse
except:
    print("ERROR: ML-SAScombine tried to import python package argparse - is it correctly installed?\n")
    import argparse
try:
    import numpy as np
except:
    print("ERROR: ML-SAScombine tried to import python package numpy - is it correctly installed?\n")
    import numpy as np
try:
    import os
except:
    print("ERROR: ML-SAScombine tried to import python package os - is it correctly installed?\n")
    import os
try:  
    import shutil
except:
    print("ERROR: ML-SAScombine tried to import python package shutil - is it correctly installed?\n")
    import shutil
try: 
    import matplotlib.pyplot as plt
except:
    print("## WARNING: ML-SAScombine tried to import python package matplotlib - is it correctly installed?\n")
    import matplotlib.pyplot as plt
try:
    from math import ceil
except:
    print("ERROR: ML-SAScombine tried to import python package ceil from math - is it correctly installed?\n")
    from math import ceil
try:
    import time
except:
    print("ERROR: ML-SAScombine tried to import python package time - is it correctly installed?\n")   
    import time
try:
    from mlsascombine_functions import *
except:
    print("ERROR: ML-SAScombine tried to import functions from files mlsascombine_functions.py")
    print("this file should be in the same directory as mlsascombine.py\n")
    from mlsascombine_functions import *

if __name__ == "__main__":
    
    t_start = time.time()

    ## input values

    # presentation
    parser = argparse.ArgumentParser(description="""ML-SAScombine - combining SAXS or SANS data using maximum likelihood""",usage="python mlsascombine.py -d \"data1.dat data2.dat data3.dat\" <OPTIONAL ARGUMENTS>" )

    # options with input
    parser.add_argument("-d", "--data", help="Datafiles (format: \"d1.dat d2.dat\"). Include path and file extension or use --path and --ext flags.")
    parser.add_argument("-p", "--path", help="Add this path to all data", default="./")
    parser.add_argument("-ext", "--ext", help="Add this extension to all data. If --data is not provided, all files with this extension will be used.", default="")
    parser.add_argument("-l", "--label", help="Labels for each datafile (separated by space)", default="none")
    parser.add_argument("-qmin", "--qmin", help="minimum q-value in combined data file", default="none")
    parser.add_argument("-qmax", "--qmax", help="maximum q-value in combined data file", default="none")
    parser.add_argument("-N", "--N", type=int, help="Maximum Number of points in combined data", default="500")
    parser.add_argument("-t", "--title", help="plot title, also used for output name [recommended]",default='Combined data')
    parser.add_argument("-ref", "--ref", help="Provide ref data (full path) for scaling - not included in combined data is not in data list. Write an integer to use a dataset from the list (e.g. 2 for dataset number 2) [default: 1].", default="none")
    parser.add_argument("-qmin_ref", "--qmin_ref", help="Provide a min q to use in reference data, for alignment [default: 0]", default="0")
    parser.add_argument("-qmax_ref", "--qmax_ref", help="Provide a max q to use in reference data, for alignment [default: no max value]", default="9999")
    parser.add_argument("-exc", "--exclude", help="Exclude one or more datasets from list. list of integers with ",default="none")
    parser.add_argument("-cc", "--conv_crit", help="Convergence criteria change of chi-square [default: 0.0001]",default="0.0001")
    parser.add_argument("-qtemp", "--q_template", help="Provide file for q template (only using first column of file) [default: no template used]", default="none")
    parser.add_argument("-qmin_all", "--qmin_all", help="Provide individual qmin values for all data (format: \"0.02 0.001\")",default="none")
    parser.add_argument("-qmax_all", "--qmax_all", help="Provide individual qmax values for all data (format: \"0.3 1.0\")",default="none")
    
    # true/false options
    parser.add_argument("-r", "--range", action="store_true", help="Only include q range with overlap of min 2 datasets",default=False)
    parser.add_argument("-rs", "--ref_smooth", action="store_true", help="Smooth reference curve before alignment [not recommended]", default=False)
    parser.add_argument("-nc", "--no_conv", action="store_true", help="Do not continue iteratively until convergence", default=False)
    parser.add_argument("-nn", "--no_normalize", action="store_true", help="Do not normalize combined dataset", default=False)
    parser.add_argument("-sc", "--output_scale", action="store_true", help="Output scale factors and constant adjustments", default=False)
    parser.add_argument("-nl", "--no_log_q", action="store_true", help="Make the combined data evenly distributed on lin scale (instead of on log scale)",default=False)
    parser.add_argument("-exp", "--export", action="store_true", help="Export scaled and subtracted curves", default=False)
    parser.add_argument("-res", "--res", action="store_true", help="Export file with residuals", default=False)
    parser.add_argument("-ft", "--ftest", action="store_true", help="Make F-test for error consistency",default=False)
    parser.add_argument("-equi", "--q_equispaced", action="store_true", help="Equispaced q (do not use weighted average for q in combined data)",default=False)
    parser.add_argument("-base", "--logbase", help="base for logarithmic rebinning (default: 1.05)",default=1.05)
    parser.add_argument("-offset2", "--offset_option2", action="store_true", help="Instead of offset to avoid negative values, high-q points are set to zero", default=False)

    # plot options
    parser.add_argument("-pa", "--plot_all", action="store_true", help="Plot all pairwise fits [for outlier analysis]", default=False)
    parser.add_argument("-pn", "--plot_none", action="store_true", help="Plot nothing", default=False)
    parser.add_argument("-pm", "--no_plot_merge", action="store_false", help="Do not plot the combined data (only the scaled datasets)", default=True)
    parser.add_argument("-err", "--error_bars", action="store_true", help="Plot errorbars in all plots [may not work well for many datasets]", default=False)
    parser.add_argument("-lin", "--plot_lin", action="store_true", help="Plot on lin-log scale (instead of log-log)", default=False)
    parser.add_argument("-sp", "--save_plot", action="store_true", help="Save pdf of plot", default=False)
    #parser.add_argument("-v", "--verbose", action="store_true", help="verbose: more output [default True]", default=True)

    args = parser.parse_args()
    
    ## read input values
    data_in = args.data
    path = args.path
    extension = args.ext
    N_merge = args.N
    PLOT_ALL   = args.plot_all
    PLOT_NONE = args.plot_none
    SAVE_PLOT  = args.save_plot
    PLOT_MERGE = args.no_plot_merge
    EXPORT = args.export
    title = args.title.replace(' ','_')
    ref_data_in = args.ref
    VERBOSE = True #VERBOSE = args.verbose
    qmin_ref = float(args.qmin_ref)
    qmax_ref = float(args.qmax_ref)
    exclude_in = args.exclude
    conv_threshold = float(args.conv_crit) 
    q_temp_data_in = args.q_template

    ## convert data string to list and remove empty entries
    try:
        data_tmp = data_in.split(' ')      
        data = []
        for i in range(len(data_tmp)):
            if not data_tmp[i] in ['',' ','  ','   ','    ','     ','      ','       ','        ']:
                data.append(data_tmp[i])
    except:
        if not extension == "":
            data = [file for file in os.listdir(path) if file.endswith(extension)]
            extension = ""
        else:
            print("ERROR: could not find data. Try with option -d \"data1.dat data2.dat\"")
            sys.exit(1)

    ## do the same for exclude input
    if not exclude_in == "none": 
        exclude_tmp = exclude_in.split(' ')
        exclude  = []
        for i in range(len(exclude_tmp)):
            if not exclude_tmp[i] in ['',' ','  ','   ','    ','     ','      ','       ','        ']:
                exclude.append(exclude_tmp[i])
        for i in range(len(exclude)):
            if exclude[i].isdigit():
                data_idx = int(exclude[i])-1
                exclude[i] = data[data_idx]
        for exc in exclude:
            if exc in data:
                data.remove(exc)
                print("excluded dataset %s" % exc)
            else:
                print("tried to exclude %s, but this data is not in list of data" % exc)
        
    if not data:
        print("ERROR: could not find data. Try with option -d \"data1.dat data2.dat\"")
        sys.exit(1)

    if len(data) == 1:
        print("ERROR: only 1 dataset, need at least 2. Try with option -d \"data1.dat data2.dat\"")
        sys.exit(1)

    ## labels
    if args.label == "none":
        labels = []
        for l in data:
            tmp = l.split('.')[0]
            labels.append(tmp.split('/')[-1])
    else:
        labels = args.label.split(' ')
    ms = 4 # markersize in plots
    
    ## determine qmin and qmax
    qmin_data,qmax_data = find_qmin_qmax(path,data,extension,args.range)
    if args.qmin == "none":
        qmin = qmin_data
    else:
        qmin = float(args.qmin)
        if qmin_data > qmin:
            qmin = qmin_data
    if args.qmax == "none":
        qmax = qmax_data
    else:
        qmax = float(args.qmax)
        if qmax_data < qmax:
            qmax = qmax_data

    ## individual qmin and qmax values
    if args.qmin_all == "none":
        qmin_all = np.ones(len(data))*qmin
    else:
        qmin_all = [float(num) for num in args.qmin_all.split(' ')]
    if args.qmax_all == "none":
        qmax_all = np.ones(len(data))*qmax
    else:
        qmax_all = [float(num) for num in args.qmax_all.split(' ')]    

    ## make q
    if not q_temp_data_in == "none":
        header,footer = get_header_footer(q_temp_data_in)
        q_temp = np.genfromtxt(q_temp_data_in,skip_header=header,skip_footer=footer,usecols=[0],unpack=True)
        N_merge = len(q_temp)
    elif args.no_log_q:
        q_temp = np.linspace(qmin,qmax,N_merge)
    else:
        q_temp = 10**np.linspace(np.log10(qmin),np.log10(qmax),N_merge)

    ## filename and folder for output
    merge_dir = 'output_%s' % title
    filename_out = '%s/merge_%s.dat' % (merge_dir,title)
    try: 
        os.mkdir(merge_dir)
    except:
        shutil.rmtree(merge_dir)
        os.mkdir(merge_dir)
        print('Output directory %s already existed - deleted old directory and created new' % merge_dir)

    ## file for stdout using printt function
    f_out = open('%s/%s_out.txt' % (merge_dir,title),'w')
    def printt(s):
        print(s)
        f_out.write('%s\n' %s)
        
    ## read reference data input
    if ref_data_in == "none":
        ref_filename = '%s%s%s' % (path,data[0],extension)
        if not os.path.exists(ref_filename):
            ref_filename = '%s/%s%s' % (path,data[0],extension)
            if not os.path.exists(ref_filename):
                ref_filename = '%s%s.%s' % (path,data[0],extension)
                if not os.path.exists(ref_filename):
                    ref_filename = '%s/%s.%s' % (path,data[0],extension)
                    if not os.path.exists(ref_filename):
                        ref_filename = '%s%s%s' % (path,data[0],extension)
        ref_data_list = [ref_filename]
    elif ref_data_in == "all":
        ref_data_list = []
        for d in data:
            ref_filename = '%s%s%s' % (path,d,extension)
            if not os.path.exists(ref_filename):
                ref_filename = '%s/%s%s' % (path,d,extension)
                if not os.path.exists(ref_filename):
                    ref_filename = '%s%s.%s' % (path,d,extension)
                    if not os.path.exists(ref_filename):
                        ref_filename = '%s/%s.%s' % (path,d,extension)
                        if not os.path.exists(ref_filename):
                            ref_filename = '%s%s%s' % (path,d,extension)
            ref_data_list.append(ref_filename) 
    elif ref_data_in.isdigit():
        number = int(ref_data_in)
        if number > len(data):
            printt('WARNING: (regarding -ref flag) No dataset number %d (obs: indexing with 1). Using first dataset as reference data' % number)
            number = 1
        idx_ref_data = number-1
        ref_filename = '%s%s%s' % (path,data[idx_ref_data],extension)
        if not os.path.exists(ref_filename):
            ref_filename = '%s/%s%s' % (path,data[idx_ref_data],extension)
            if not os.path.exists(ref_filename):
                ref_filename = '%s%s.%s' % (path,data[idx_ref_data],extension)
                if not os.path.exists(ref_filename):
                    ref_filename = '%s/%s.%s' % (path,data[idx_ref_data],extension)
                    if not os.path.exists(ref_filename):
                        ref_filename = '%s%s%s' % (path,data[idx_ref_data],extension)

        ref_data_list = [ref_filename]
    else:
        ref_data_list = [ref_data_in]
    
    ## read input command
    input_string = 'python'
    for aa in sys.argv:
        if ' ' in aa:
            input_string += " \"%s\"" % aa
        else:
            input_string += " %s" %aa

    ## welcome message
    printt('#########################################')
    printt('RUNNING mlsascombine.py, version %s \nfor instructions: python mlsascombine.py -h' % version)
    printt('command used: python %s' % input_string)
    printt('#########################################')

    ## print input values
    printt('data:')
    for name in data:
        printt('       %s' % name)
    printt('qmin: %f' % qmin)
    printt('qmax: %f' % qmax)
    if not q_temp_data_in == "none":
        printt('q template: %s' % q_temp_data_in)
    else:
        printt('N_max: %d' % N_merge)
    printt('ref: %s' % ref_data_list[0])
    if not args.no_conv:
        imax = 30
        for i in range(imax+1):
            ref_data_list.append(filename_out)
        EXPORT = False
        PLOT_NONE = True
        PLOT_ALL = False
        SAVE_PLOT = False
        PLOT_MERGE = False
        STOP_NEXT = False
        VERBOSE = False
        printt('The results are independent on the choise of reference curve, unless --no_conv is used')

    if args.res:
       res_dir = '%s/residuals' % merge_dir
       os.mkdir(res_dir)

    ## loop over reference data list until you get converged solution
    count = 0
    for ref_data in ref_data_list:

        ## initialize figure
        if not PLOT_NONE:
            fig,ax = plt.subplots(figsize=(12,6))

        ## import reference data
        header,footer = get_header_footer(ref_data)
        q_ref,I_ref = np.genfromtxt(ref_data,skip_header=header,skip_footer=footer,usecols=[0,1],unpack=True)
        if qmin_ref != 0 or qmax_ref != 9999:
            idx = np.where((q_ref <= qmax_ref) & (q_ref >= qmin_ref))
            q_ref,I_ref = q_ref[idx],I_ref[idx]

        if args.ref_smooth:
            N_sm = np.amax([ceil(len(q_ref)/50),1])
            I_ref = smooth(I_ref,N_sm,'lin')

        if EXPORT:
            exp_dir = '%s/scaled_data' % merge_dir
            try: 
                os.mkdir(exp_dir)
            except:
                shutil.rmtree(exp_dir)
                os.mkdir(exp_dir)
                printt('Output directory %s already existed - delete old directory and created new' % exp_dir)

        ## combine data
        q_sum,I_sum,w_sum = np.zeros(N_merge),np.zeros(N_merge),np.zeros(N_merge)
        if args.ftest:
            q_matrix,I_matrix,dI_matrix,w_matrix = [[] for x in range(N_merge)],[[] for x in range(N_merge)],[[] for x in range(N_merge)],[[] for x in range(N_merge)]
        chi2r_list = []
        if args.output_scale:
            a_list,b_list = [],[]
        for datafile,label,qmin_i,qmax_i in zip(data,labels,qmin_all,qmax_all):
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
            q_in,I_in,dI_in = np.genfromtxt(filename,skip_header=header,skip_footer=footer,usecols=[0,1,2],unpack=True)
            qmin_global = np.amax([qmin,qmin_i])
            qmax_global = np.amin([qmax,qmax_i])
            idx = np.where((q_in >= qmin_global) & (q_in <= qmax_global))
            q,I,dI = q_in[idx],I_in[idx],dI_in[idx]
            M = len(q)            

            ## truncate data (only for scaling), to have same q-range as ref data
            q_t,I_t,dI_t = trunc(q,I,dI,q_ref)

            ## fit ref data to truncated data 
            I_interp_fit,popt = fit_scale_offset(q_t,I_t,dI_t,q_ref,I_ref)

            ## calc residuals and chi2r
            R = (I_t - I_interp_fit)/dI_t
            dof = len(q_t)-2 # 2 is the number of fitting parameters len(popt)
            chi2r,p = calculate_chi2r(R,dof)
            chi2r_list.append(chi2r)

            ## get scaling and offset
            a,b = popt
            I_fit = (I-b)/a
            I_in_fit = (I_in-b)/a
            dI_fit = dI/a 
            dI_in_fit = dI_in/a 
            if args.output_scale:
                a_list.append(1/a)
                b_list.append(-b/a) 

            if args.no_conv:
                fit_data = 'reference data'
            else:
                fit_data = 'combined data'

            ## plot interpolation
            if PLOT_ALL and not PLOT_NONE:
                figa,axa = plt.subplots(2,1,gridspec_kw={'height_ratios': [4,1]},figsize=(10,10))

                axa[0].errorbar(q,I,yerr=dI,marker='.',markersize=ms,linestyle='none',zorder=0,label=label)
                axa[0].plot(q_t,I_interp_fit,color='black',label=r'fit with %s, $\chi^2$ %1.1f' % (fit_data,chi2r),zorder=1)
                axa[0].set_yscale('log')
                axa[0].set_xlabel('q')
                axa[0].set_ylabel('Intensity')
                axa[0].legend()
                axa[0].set_title('fit of %s data to %s' % (fit_data,label))

                Rmax = ceil(np.amax(abs(R)))
                axa[1].plot(q_t,0*q_t,color='black',zorder=1)
                axa[1].plot(q_t,R,linestyle='none',marker='.',markersize=ms,zorder=0)
                axa[1].set_ylim(-Rmax,Rmax)
                if Rmax >= 4:
                    axa[1].plot(q_t,3*np.ones(len(q_t)),color='grey',linestyle='--',zorder=1)
                    axa[1].plot(q_t,-3*np.ones(len(q_t)),color='grey',linestyle='--',zorder=1)
                    axa[1].set_yticks([-Rmax,-3,0,3,Rmax])
                else:
                    axa[1].set_yticks([-Rmax,0,Rmax])
                axa[1].set_xlim(axa[0].get_xlim())

                if SAVE_PLOT:
                    figa.savefig('%s/fit_%s' % (merge_dir,label))
            
            ## export residuals etc
            if args.res:
                with open('%s/fit_res_%s.dat' % (res_dir,label),'w') as f:
                    f.write('# fit %s with %s, chi2r = %1.2f\n' % (label,fit_data,chi2r))
                    f.write('%-14s %-14s %-14s %-14s %-20s\n' % ('# q','  I','  sigma','  Ifit','  R = (I-Ifit)/sig'))
                    for x1,x2,x3,x4,x5 in zip(q_t,I_t,dI_t,I_interp_fit,R):
                        f.write('%14e %14e %14e %14e %14e\n' % (x1,x2,x3,x4,x5))

            ## export data
            if EXPORT:
                filename_scaled = "%s/%s_scaled.dat" % (exp_dir,label)
                with open(filename_scaled,'w') as f:
                    f.write('scaled and subtraced version of %s\n' % filename)
                    f.write('scaled to align with %s\n' % ref_data)
                    f.write('# q  I  sigma\n')
                    for  q_i,I_i,dI_i in zip(q,I_fit,dI_fit):
                        f.write('%e %e %e\n' % (q_i,I_i,dI_i))
                    
            add_data(q_sum,I_sum,w_sum,q,I_fit,dI_fit,q_temp)
            if args.ftest: 
                append_data(q_matrix,I_matrix,dI_matrix,w_matrix,q,I_fit,dI_fit,q_temp)

            ## plot data
            if not PLOT_NONE:
                if args.error_bars:
                    ax.errorbar(q_in,I_in_fit,yerr=dI_in_fit,marker='.',markersize=ms,linestyle='none',label=label,zorder=0)
                else:
                    ax.plot(q,I_fit,marker='.',markersize=ms,linestyle='none',label=label,zorder=1)
                    ax.plot(q_in,I_in_fit,marker='.',markersize=ms,linestyle='none',zorder=0,color=ax.get_lines()[-1].get_color(),alpha=0.2)

            ## output
            if VERBOSE:
                printt('----------------------------------------------------------------\ncompare %s with %s\n----------------------------------------------------------------' % (label,ref_data))
                printt('N of %s: %d' % (label,M))
                printt('chi2r = %1.1f (dof=%d, p=%1.6f)' % (chi2r,dof,p))
                if p < 0.0001:
                    printt("WARNING: data may be incompatible (p<0.0001). Rerun with flag --plot_all for visual comparison and residuals")
                if EXPORT:
                    printt('Scaled and subtracted data written to file: %s' % filename_scaled)

        ## weighted averages
        idx = np.where(w_sum>0.0)   
        if args.q_equispaced or not q_temp_data_in == "none":
            q_merge = q_temp[idx]
        else:
            q_merge = q_sum[idx]/w_sum[idx]
        I_merge = I_sum[idx]/w_sum[idx]
        dI_merge = w_sum[idx]**-0.5

        if args.ftest:
            F_c = 20 # critical F value
            count_err,count_fine,j = 0,0,0
            for i in range(n):
                ni = len(q_matrix[i])
                if ni != 0:
                    sd = np.std(I_matrix[i])
                    se = sd/np.sqrt(ni)
                    sig = np.sqrt(np.mean(np.array(dI_matrix[i])**2))
                    sig_mean = sig/np.sqrt(ni)
                    sig_ml = w_sum[i]**-0.5 # sigma using maximum likelihood
                    qi = np.mean(np.array(q_matrix[i]))
                    F = se/sig_ml
                    if F > F_c:
                        if VERBOSE:
                            printt('WARNING: data at q: %1.1e, may not be compatible as std_error/maximum_likelihood_error = %1.2f > %d)' %(qi,F,F_c))
                        #dI_merge[j] = sig_mean
                        count_err += 1
                    else:
                        count_fine += 1
                    j += 1

            if count_err > 0 and VERBOSE:
                printt('Number of points with very large (more than x' + str(F_c) + ' larger) standard error compared to maximum likelihood error: ' + str(count_err) + ', points with OK errors: ' + str(count_fine))
                #printt('Using sum of squares error propagation instead of maximum likelihood error propagation for the points with too small error')
   
        if not PLOT_NONE:
            if PLOT_MERGE:
                if args.error_bars:
                    ax.errorbar(q_merge,I_merge,yerr=dI_merge,linestyle='none',marker='.',markersize=ms,color='black',zorder=1,label='Combined data')
                else:
                    ax.plot(q_merge,I_merge,linestyle='none',marker='.',markersize=ms,color='black',zorder=1,label='Combined data')

        if not args.no_normalize:
            ## normalize before export
            if args.offset_option2:
                last = int(0.02*len(I_merge)) # last 2%
                offset = np.mean(I_merge[-last:]) # average of last points
            else:
                offset = np.min(I_merge) # ensures all points are positive
            #I_sort = np.sort(I_merge) # sort the array to find the 10th lowest point
            #offset = I_sort[10]
            I_merge -= offset 
            I_merge += 1e-3 # add a constant
            I0 = np.mean(I_merge[0:4])
            I_merge /= I0
            dI_merge /= I0
        
        with open(filename_out,'w') as f:
            f.write('# sample: %s\n' % args.title)
            f.write('# data\n')
            for dataname in data:
                f.write('# %s\n' % (dataname))
            f.write('# q  I  sigma\n')
            for (qi,Ii,dIi) in zip(q_merge,I_merge,dI_merge):
                f.write('%e %e %e\n' % (qi,Ii,dIi))

        ## figure settings
        if not PLOT_NONE:
            ax.set_title(args.title)
            if not args.plot_lin:
                ax.set_xscale('log')
                xmin = qmin * 0.5
            else:
                xmin = 0
            ax.set_yscale('log')
            ax.set_xlabel('q')
            ax.set_ylabel('Intensity')
            ax.legend(bbox_to_anchor=(1.3,1.0))
            ax.set_xlim(xmin,qmax*1.2)
            fig.tight_layout()
            if SAVE_PLOT:
                fig.savefig('%s/merge_%s' % (merge_dir,title)) 

        try:
            if chi2r_list_prev is not None:
                STOP = True
                for (c,cp) in zip(chi2r_list,chi2r_list_prev):
                    if abs(c-cp) > conv_threshold:
                        STOP = False
            if ref_data == filename_out:
                count += 1
        except: 
            STOP = False

        if not args.no_conv:
            if STOP_NEXT:
                printt('#########################################')
                printt('Converged after %d iterations' % count)
                printt('N in combined data: %d' % len(idx[0]))
                if args.range:
                    printt('q range with at least 2 overlapping data curves: [%1.4f,%1.2f]' % (qmin,qmax))
                printt('Combined data written to file: %s' % filename_out)
                if qmin_ref != 0 or qmax_ref != 9999:
                    printt('Data sorted after compatibility with combined consensus curve, in selected q-range (--qmin_ref and qmax_ref):')
                else:
                    printt('Data sorted after compatibility with combined consensus curve:')
                    printt('%20s  %s' % ('name of datafile','reduced chi-square'))
                for i in np.argsort(chi2r_list):
                    if args.output_scale:
                        printt('%20s: %1.2f (a=%1.3f, b=%1.6f)' % (data[i],chi2r_list[i],a_list[i],b_list[i]))
                    else:
                        printt('%20s: %1.2f' % (data[i],chi2r_list[i]))
                end_time =  time.time() - t_start
                printt('%20s: %1.2f' % ('sum',np.sum(chi2r_list)))
                printt('#########################################')
                printt('ML-SAScombine finished successfully')
                printt('output sent to folder %s' % merge_dir)
                printt("run time: %1.2f" % end_time)
                printt('#########################################')    
                f_out.close()
                if not PLOT_NONE:
                    plt.show()
                sys.exit(0)
            if STOP:    
                PLOT_ALL = args.plot_all
                PLOT_NONE = args.plot_none
                SAVE_PLOT = args.save_plot
                PLOT_MERGE = args.no_plot_merge
                EXPORT = args.export
                VERBOSE = True
                STOP_NEXT = True
            else: 
                chi2r_list_prev = chi2r_list
                if count >= imax:
                    PLOT_ALL   = args.plot_all
                    PLOT_NONE = args.plot_none
                    SAVE_PLOT  = args.save_plot
                    PLOT_MERGE = args.no_plot_merge
                    EXPORT = args.export
                    VERBOSE = True

        ## output
        if VERBOSE:
            if not args.no_conv:
                if count > imax:
                    printt('#########################################')
                    printt('Max number of iterations reached (imax = %d)' % imax)
                    printt('N in combined data: %d' % len(idx[0]))
                    if args.range:
                        printt('q range with at least 2 overlapping data curves: [%1.4f,%1.2f]' % (qmin,qmax))
                    printt('Combined data written to file: %s' % filename_out)
                    if qmin_ref != 0 or qmax_ref != 9999:
                        printt('Data sorted after compatibility with combined consensus curve, in selected q-range (--qmin_ref and qmax_ref):')
                    else:
                        printt('Data sorted after compatibility with combined consensus curve:')
                        printt('Data   chi2r')
                    for i in np.argsort(chi2r_list):
                        if args.output_scale:
                            printt('%20s: %1.2f (a=%1.3f, b=%1.6f)' % (data[i],chi2r_list[i],a_list[i],b_list[i]))
                        else:
                            printt('%20s: %1.2f' % (data[i],chi2r_list[i]))
                    printt('%20s: %1.2f' % ('sum',np.sum(chi2r_list)))
            else:
                    printt('#########################################')
                    printt('N in combined data: %d' % len(idx[0]))
                    printt('Combined data written to file: %s' % filename_out)

    end_time =  time.time() - t_start   
    printt('#########################################')
    printt('ML-SAScombine.py finished successfully')
    printt('output sent to folder %s' % merge_dir)
    printt("run time: %1.2f" % end_time)
    printt('#########################################')
    f_out.close()
    if not PLOT_NONE:
        plt.show()
