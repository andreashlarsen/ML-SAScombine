# SASmerge

SASmerge merges SAS data into a merged consensus dataset   
e.g. data measured on different instruments on the same sample or SAXS and WAXS data    

## Download
sasmerge is a python3 program, so you need python3    
download sasmerge.py and the other python scripts in this repository       
see dependencies  

## Run  

### standard run for merging 3 datasets
python sasmerge.py --data "dataset1.dat dataset2.dat dataset3.dat" --title my_merged_data

### for instructions and options, type: 
python sasmerge --help

## Dependencies

### dependencies from this folder:     
* smooth.py    
* find_qmin_qmax.py    
* add_data.py      
* calculate_chi2r.py    
* get_header_footer.py     

### other dependencies (standard python packages):   
* argparse     
* numpy    
* matplotlib    
* os    
* shutil    
* math    
* scipy    

## Notes  and warnings
* the program is a beta version - so use it with care. But feedback is more than welcome    
* sasmerge can be applied to SANS data, but resolution effects are not taken into account, which will inevitably lead to systematic errors   

## Credit
The program was written by Andreas Haahr Larsen   
Input and insight from Jochen Hub, Jill Trewhella and Patrice Vachette   
