
###########################
# SASmerge version beta0.4
###########################

## importing python packages
import sys
try:
    import argparse
except:
    print("ERROR: sasmerge tried to import python package argparse - is it correctly installed?\n")
    sys.exit(1)
try:
    import numpy as np
except:
    print("ERROR: sasmerge tried to import python package numpy - is it correctly installed?\n")
    sys.exit(1)
try:
    import os
except:
    print("ERROR: sasmerge tried to import python package os - is it correctly installed?\n")
    sys.exit(1)
try:  
    import shutil
except:
    print("ERROR: sasmerge tried to import python package shutil - is it correctly installed?\n")
    sys.exit(1)
try: 
    import matplotlib.pyplot as plt
except:
    print("## WARNING: sasmerge tried to import python package matplotlib - is it correctly installed?\n")
    sys.exit(1)
try:
    from math import ceil
except:
    print("ERROR: sasmerge tried to import python package ceil from math - is it correctly installed?\n")
    sys.exit(1)
try:
    import time
except:
    print("ERROR: sasmerge tried to import python package time - is it correctly installed?\n")   
try:
    from get_header_footer import get_header_footer as ghf
    from find_qmin_qmax import find_qmin_qmax 
    from add_data import add_data
    from smooth import smooth
    from calculate_chi2r import *
except:
    print("ERROR: sasmerge tried to import functions from files get_header_footer.py find_qmin_qmax.py add_data.py smooth.py calculate_chi2r.py")
    print("these files should be in the same directory as sasmerge.py\n")
    sys.exit(1)

if __name__ == "__main__":
    
    t_start = time.time()

    ## input values

    # presentation
    parser = argparse.ArgumentParser(description="""SASmerge - merge multiple SAXS or SANS datasets""",usage="python sasmerge.py -d \"data1.dat data2.dat data3.dat\" <OPTIONAL ARGUMENTS>" )

    # options with input
    parser.add_argument("-d", "--data", help="Datafiles (format: \"d1.dat d2.dat\"). Include path and file extension or use --path and --ext flags.")
    parser.add_argument("-p", "--path", help="Add this path to all data", default="./")
    parser.add_argument("-ext", "--ext", help="Add this extension to all data. If --data is not provided, all files with this extension will be used.", default="")
    parser.add_argument("-l", "--label", help="Labels for each datafile (separated by space)", default="none")
    parser.add_argument("-qmin", "--qmin", help="minimum q-value in merged file", default="none")
    parser.add_argument("-qmax", "--qmax", help="maximum q-value in merged file", default="none")
    parser.add_argument("-N", "--N", type=int, help="Maximum Number of points in merged data", default="500")
    parser.add_argument("-t", "--title", help="plot title, also used for output name [recommended]",default='Merged')
    parser.add_argument("-ref", "--ref", help="Provide ref data (full path) for scaling - not included in merged data is not in data list. Write an integer to use a dataset from the list (e.g. 2 for dataset number 2) [default: 1].", default="none")
    parser.add_argument("-qmin_ref", "--qmin_ref", help="Provide a min q to use in reference data, for alignment [default: 0]", default="0")
    parser.add_argument("-qmax_ref", "--qmax_ref", help="Provide a max q to use in reference data, for alignment [default: no max value]", default="9999")
    parser.add_argument("-exc", "--exclude", help="Exclude one or more datasets from list. list of integers with ",default="none")
    parser.add_argument("-cc", "--conv_crit", help="Convergence criteria change of chi-square [default: 0.0001]",default="0.0001")
    
    # true/false options
    parser.add_argument("-r", "--range", action="store_true", help="only include q range with overlap of min 2 datasets",default=False)
    parser.add_argument("-rs", "--ref_smooth", action="store_true", help="smooth reference curve before alignment [not recommended]", default=False)
    parser.add_argument("-nc", "--no_conv", action="store_false", help="do not continue iteratively until convergence", default=True)
    parser.add_argument("-nn", "--no_normalize", action="store_false", help="do not normalize merged dataset", default=True)
    parser.add_argument("-sc", "--output_scale", action="store_true", help="output scale factors and constant adjustments", default=False)
    parser.add_argument("-nl", "--no_log_q", action="store_false", help="make the merged data equispaced on lin-scale (instead of on log-scale which is default)",default=True)
    parser.add_argument("-exp", "--export", action="store_true", help="export scaled and subtracted curves", default=False)
    
    # plot options
    parser.add_argument("-pa", "--plot_all", action="store_true", help="Plot all pairwise fits [for outlier analysis]", default=False)
    parser.add_argument("-pn", "--plot_none", action="store_true", help="Plot nothing", default=False)
    parser.add_argument("-pm", "--no_plot_merge", action="store_false", help="Do not plot the merged data (only the scaled datasets)", default=True)
    parser.add_argument("-err", "--error_bars", action="store_true", help="plot errorbars in all plots [may not work well for many datasets]", default=False)
    parser.add_argument("-lin", "--plot_lin", action="store_true", help="plot on lin-log scale (instead of log-log)", default=False)
    parser.add_argument("-sp", "--save_plot", action="store_true", help="Save pdf of plot", default=False)
    #parser.add_argument("-v", "--verbose", action="store_true", help="verbose: more output [default True]", default=True)

    args = parser.parse_args()
    
    ## read input values
    data_in = args.data
    path = args.path
    extension = args.ext
    label_in = args.label
    qmin_in = args.qmin
    qmax_in = args.qmax
    N_merge = args.N
    LOG_Q = args.no_log_q
    PLOT_ALL   = args.plot_all
    PLOT_NONE = args.plot_none
    SAVE_PLOT  = args.save_plot
    PLOT_MERGE = args.no_plot_merge
    EXPORT = args.export
    PLOT_LIN = args.plot_lin
    title = args.title
    RANGE = args.range
    ref_data_in = args.ref
    SMOOTH = args.ref_smooth
    CONV = args.no_conv
    ERRORBAR = args.error_bars
    VERBOSE = True #VERBOSE = args.verbose
    NORM = args.no_normalize
    SCALE_OUTPUT = args.output_scale
    qmin_ref = float(args.qmin_ref)
    qmax_ref = float(args.qmax_ref)
    exclude_in = args.exclude
    conv_threshold = float(args.conv_crit) 

    ## convert data string to list and remove empty entries
    try:
        data_tmp = data_in.split(' ')      
        data = []
        for i in range(len(data_tmp)):
            if not data_tmp[i] in ['',' ','  ','   ','    ','     ','      ','       ','        ']:
                data.append(data_tmp[i])
    except:
        data = [file for file in os.listdir(path) if file.endswith(extension)]
        extension = ""
        

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

    ## labels
    if label_in == "none":
        labels = []
        for l in data:
            labels.append(l.split('.')[0])
    else:
        labels = label_in.split(' ')
    ms = 4 # markersize in plots

    ## determine qmin and qmax
    qmin_data,qmax_data = find_qmin_qmax(path,data,extension)
    if qmin_in == "none":
        qmin = qmin_data
    else:
        qmin = float(qmin_in)
        if qmin_data > qmin:
            qmin = qmin_data
    if qmax_in == "none":
        qmax = qmax_data
    else:
        qmax = float(qmax_in)
        if qmax_data < qmax:
            qmax = qmax_data

    ## make q
    if LOG_Q:
        q_edges = 10**np.linspace(np.log10(qmin),np.log10(qmax),N_merge+1)
    else:
        q_edges = np.linspace(qmin,qmax,N_merge+1)

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

    ## print input values
    printt('#########################################')
    printt('RUNNING sasmerge.py \nfor instructions: python sasmerge.py -h')
    printt('#########################################')
    printt('data :')
    for name in data:
        printt('       %s' % name)
    printt('qmin : %f' % qmin)
    printt('qmax : %f' % qmax)
    printt('N_max: %d' % N_merge)
    printt('ref  : %s' % ref_data_list[0])
    if CONV:
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

    ## loop over reference data list until you get converged solution
    count = 0
    for ref_data in ref_data_list:

        ## initialize figure
        #if not (PLOT_NONE or PLOT_ALL):
        if not PLOT_NONE:
            fig,ax = plt.subplots(figsize=(10,10))

        ## import reference data
        header,footer = ghf(ref_data)
        q_ref,I_ref = np.genfromtxt(ref_data,skip_header=header,skip_footer=footer,usecols=[0,1],unpack=True)
        if qmin_ref != 0 or qmax_ref != 9999:
            idx = np.where((q_ref <= qmax_ref) & (q_ref >= qmin_ref))
            q_ref,I_ref = q_ref[idx],I_ref[idx]

        if SMOOTH:
            N_sm = np.amax([ceil(len(q_ref)/50),1])
            I_ref = smooth(I_ref,N_sm,'lin')

        if EXPORT:
            exp_dir = 'output_%s/scaled_data' % title
            try: 
                os.mkdir(exp_dir)
            except:
                shutil.rmtree(exp_dir)
                os.mkdir(exp_dir)
                printt('Output directory %s already existed - delete old directory and created new' % exp_dir)

        ## merge data
        q_sum,I_sum,w_sum = np.zeros(N_merge),np.zeros(N_merge),np.zeros(N_merge)
        chi2r_list = []
        if RANGE:
            qmin_list,qmax_list = [],[]
        if SCALE_OUTPUT:
            a_list,b_list = [],[]
        for datafile,label in zip(data,labels):
            filename = '%s%s%s' % (path,datafile,extension)
            if not os.path.exists(filename):
                filename = '%s/%s%s' % (path,datafile,extension)
                if not os.path.exists(filename):
                    filename = '%s%s.%s' % (path,datafile,extension)
                    if not os.path.exists(filename):
                        filename = '%s/%s.%s' % (path,datafile,extension)
                        if not os.path.exists(filename):
                            filename = '%s%s%s' % (path,datafile,extension)
            header,footer = ghf(filename)
            q_in,I_in,dI_in = np.genfromtxt(filename,skip_header=header,skip_footer=footer,unpack=True)
            if RANGE:
                qmin_list.append(np.amin(q_in))
                qmax_list.append(np.amax(q_in))
            idx = np.where(q_in<=qmax)
            q,I,dI = q_in[idx],I_in[idx],dI_in[idx]
            M = len(q)            

            ## truncate data (only for scaling), to have same q-range as ref data
            q_t,I_t,dI_t = trunc(q,I,dI,q_ref)

            ## fit ref data to truncated) data 
            I_interp_fit,popt = fit_scale_offset(q_t,I_t,dI_t,q_ref,I_ref)

            ## calc residuals and chi2r
            R = (I_t - I_interp_fit)/dI_t
            dof = len(q_t)-2 # 2 is the number of fitting parameters len(popt)
            chi2r,p = calculate_chi2r(R,dof)
            chi2r_list.append(chi2r)

            ## get scaling and offset
            a,b = popt
            I_fit = (I-b)/a
            dI_fit = dI/a  
            if SCALE_OUTPUT:
                a_list.append(1/a)
                b_list.append(-b/a) 

            ## plot interpolation
            if PLOT_ALL and not PLOT_NONE:
                if CONV:
                    fit_data = 'merged data'
                else:
                    fit_data = 'reference data'

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
                    figa.savefig('output_%s/fit_%s' % (title,label))

            ## export data
            if EXPORT:
                filename_scaled = "%s/%s_scaled.dat" % (exp_dir,datafile)
                with open(filename_scaled,'w') as f:
                    f.write('scaled and subtraced version of %s\n' % filename)
                    f.write('scaled to align with %s\n' % ref_data)
                    f.write('# q  I  sigma\n')
                    for  q_i,I_i,dI_i in zip(q,I_fit,dI_fit):
                        f.write('%e %e %e\n' % (q_i,I_i,dI_i))
                    
            add_data(q_sum,I_sum,w_sum,q,I_fit,dI_fit,q_edges)

            #if not (PLOT_ALL or PLOT_NONE):
            if not PLOT_NONE:
                if ERRORBAR:
                    ax.errorbar(q,I_fit,yerr=dI_fit,marker='.',markersize=ms,linestyle='none',label=label,zorder=0)
                else:
                    ax.plot(q,I_fit,marker='.',markersize=ms,linestyle='none',label=label,zorder=0)

            ## output
            if VERBOSE:
                printt('------------------------------------------------------------\ncompare %s with %s\n------------------------------------------------------------' % (label,ref_data))
                printt('N of %s: %d' % (label,M))
                printt('chi2r = %1.1f (dof=%d, p=%1.6f)' % (chi2r,dof,p))
                if p < 0.0001:
                    printt("WARNING: data may be incompatible (p<0.0001). Rerun with flag --plot_all for visual comparison and residuals")
                if EXPORT:
                    printt('Scaled and subtracted data written to file: %s' % filename_scaled)

        if RANGE:
            # sort qmin and qmax lists
            qmin_list.sort()
            qmax_list.sort()
            # find qmin and qmax with at least 2 datasets
            qmin_global,qmax_global = qmin_list[1],qmax_list[-2]

        ## weighted averages
        idx = np.where(w_sum>0.0)   
        q_merge = q_sum[idx]/w_sum[idx]
        I_merge = I_sum[idx]/w_sum[idx]
        dI_merge = w_sum[idx]**-0.5

        ## truncate to qmin and qmax with at least 2 datasets
        if RANGE:
            idx = np.where((q_merge >= qmin_global) & (q_merge <= qmax_global))
            dI_merge = dI_merge[idx]
            q_merge = q_merge[idx]
            I_merge = I_merge[idx]

        #if not (PLOT_ALL or PLOT_NONE):
        if not PLOT_NONE:
            if PLOT_MERGE:
                if ERRORBAR:
                    ax.errorbar(q_merge,I_merge,yerr=dI_merge,linestyle='none',marker='.',markersize=ms,color='black',zorder=1,label='Merged')
                else:
                    ax.plot(q_merge,I_merge,linestyle='none',marker='.',markersize=ms,color='black',zorder=1,label='Merged')

        if NORM:
            ## normalize before export
            I_merge = I_merge - np.min(I_merge) + 1e-4 #ensures all points are positive.
            I0 = np.mean(I_merge[0:4])
            I_merge /= I0
            dI_merge /= I0
        
        with open(filename_out,'w') as f:
            f.write('#sample: %s\n' % title)
            f.write('# data\n')
            for dataname in data:
                f.write('# %s\n' % (dataname))
            f.write('# q  I  sigma\n')
            for (qi,Ii,dIi) in zip(q_merge,I_merge,dI_merge):
                f.write('%e %e %e\n' % (qi,Ii,dIi))

        ## figure settings
        #if not (PLOT_ALL or PLOT_NONE):
        if not PLOT_NONE:
            ax.set_title(title)
            if not PLOT_LIN:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('q')
            ax.set_ylabel('Intensity')
            ax.legend(bbox_to_anchor=(1.3,1.0))
            fig.tight_layout()
            if SAVE_PLOT:
                fig.savefig('output_%s/merge_%s' % (title,title)) 

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

        if CONV:
            if STOP_NEXT:
                printt('#########################################')
                printt('Converged after %d iterations' % count)
                printt('N in merged data: %d' % len(idx[0]))
                if RANGE:
                    printt('q range with at least 2 overlapping data curves: [%1.4f,%1.2f]' % (qmin_global,qmax_global))
                printt('Merged data written to file: %s' % filename_out)
                if qmin_ref != 0 or qmax_ref != 9999:
                    printt('Data sorted after compatibility with merged consensus curve, in selected q-range (--qmin_ref and qmax_ref):')
                else:
                    printt('Data sorted after compatibility with merged consensus curve:')
                for i in np.argsort(chi2r_list):
                    if SCALE_OUTPUT:
                        printt('%20s: %1.2f (a=%1.3f, b=%1.6f)' % (data[i],chi2r_list[i],a_list[i],b_list[i]))
                    else:
                        printt('%20s: %1.2f' % (data[i],chi2r_list[i]))
                end_time =  time.time() - t_start
                printt('%20s: %1.2f' % ('sum',np.sum(chi2r_list)))
                printt('#########################################')
                printt('sasmerge finished successfully')
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
            if CONV:
                if count > imax:
                    printt('#########################################')
                    printt('Max number of iterations reached (imax = %d)' % imax)
                    printt('N in merged data: %d' % len(idx[0]))
                    if RANGE:
                        printt('q range with at least 2 overlapping data curves: [%1.4f,%1.2f]' % (qmin_global,qmax_global))
                    printt('Merged data written to file: %s' % filename_out)
                    if qmin_ref != 0 or qmax_ref != 9999:
                        printt('Data sorted after compatibility with merged consensus curve, in selected q-range (--qmin_ref and qmax_ref):')
                    else:
                        printt('Data sorted after compatibility with merged consensus curve:')
                        printt('Data   chi2r')
                    for i in np.argsort(chi2r_list):
                        if SCALE_OUTPUT:
                            printt('%20s: %1.2f (a=%1.3f, b=%1.6f)' % (data[i],chi2r_list[i],a_list[i],b_list[i]))
                        else:
                            printt('%20s: %1.2f' % (data[i],chi2r_list[i]))
                    printt('%20s: %1.2f' % ('sum',np.sum(chi2r_list)))
            else:
                    printt('#########################################')
                    printt('N in merged data: %d' % len(idx[0]))
                    printt('Merged data written to file: %s' % filename_out)

    end_time =  time.time() - t_start   
    printt('#########################################')
    printt('sasmerge.py finished successfully')
    printt('output sent to folder %s' % merge_dir)
    printt("run time: %1.2f" % end_time)
    printt('#########################################')
    f_out.close()
    if not PLOT_NONE:
        plt.show()
