# sasmerge

sasmerge merges SAXS* data into a merged consensus dataset   
e.g. data measured on different instruments on the same sample or SAXS and WAXS data
* can also be applied to SANS, but resolution effects are not taken into account, so this will lead to systematic errors   

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
smooth.py    
find_qmin_qmax.py    
add_data.py      
calculate_chi2r.py    
get_header_footer.py     

### other dependencies (standard python packages):   
argparse     
numpy    
matplotlib    
os    
shutil    
math    
scipy    

## credit
the program was written by Andreas Haahr Larsen   
input and insight from Jochen Hub, Jill Trewhella and Patrice Vachette   
