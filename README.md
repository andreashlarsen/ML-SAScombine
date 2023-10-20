# sasmerge

## description
merge sas data into consensus data
e.g. data measured on different instruments on the same sample or SAXS and WAXS data

## download
sasmerge is a python3 program, so you need python3    
download sasmerge.py and the other python scripts in this repository       
see dependencies  

## run  

### standard run for merging 3 datasets
python sasmerge.py --data "dataset1.dat dataset2.dat dataset3.dat" --title my_merged_data

### for instrucions, type: 
python sasmerge --help

## dependencies

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
