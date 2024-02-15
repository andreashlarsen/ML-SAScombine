#!/usr/bin/python3

import json
import io
import sys
import os
from genapp3 import genapp
import numpy as np
import time
import subprocess

if __name__=='__main__':

    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)

    ## read Json input
    datas = json_variables['data'] # names of datafile
    data_option = " -d \""
    for d in datas:
        datafile = d.split('/')[-1]
        data_option += ' ' + datafile
        path_for_output = d.replace(datafile,'')
        path_option = " -p " + path_for_output
    data_option += "\""
    qmin = json_variables['qmin']
    qmax = json_variables['qmax']
    N = json_variables['N']
    title_in = json_variables['title']
    title = title_in.replace(' ','_')

    save_option = '-sp -exp'

    ## read checkboxes and related input
    # the Json input for checkboxes only exists if boxes are checked
    # therefore I use try-except to import
    options = '-t %s -N %s -qmin %s -qmax %s' % (title,N,qmin,qmax)
    try:
        dummy = json_variables["range"]
        options += ' -r'
    except:
        pass

    try:
        dummy = json_variables['error']
        options += ' -err'
    except:
        pass

    try:
        dummy = json_variables['plot_lin']
        options += ' -lin'
    except:
        pass
    try:
        dummy = json_variables['nl']
        options += ' -nl'
    except:
        pass
    try:
        dummy = json_variables['res']
        options += ' -res'
    except:
        pass
    try:
        dummy = json_variables['pa']
        options += ' -pa'
    except:
        pass

    ## get output folder
    folder = json_variables['_base_directory'] # output folder dir

    ## messaging
    d = genapp(json_variables)

    ## run ML-SAScombine
    command = 'python3 mlsascombine.py %s %s %s' % (save_option,data_option,options)
    d.udpmessage({"_textarea":"%s\n\n" % command})
    path = os.path.dirname(os.path.realpath(__file__))
    os.system('python3 %s/mlsascombine.py %s %s %s %s' % (path,save_option,path_option,data_option,options))
    combine_dir = 'output_%s' % title
    f = open('%s/%s_out.txt' % (combine_dir,title))
    lines = f.readlines()
    for line in lines:
        d.udpmessage({"_textarea": line})
    f.close()

    ## compress output files to zip file
    os.system('zip -r results_%s.zip %s/*' % (title,combine_dir))

    ## generate output
    output = {} # create an empty python dictionary
    output["combinefig"] = "%s/%s/merge_%s.png" % (folder,combine_dir,title)
    output["zip"] = "%s/results_%s.zip" % (folder,title)

    ## send output
    print( json.dumps(output) ) # convert dictionary to json and output

