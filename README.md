# SAScombine
version beta0.9

SAScombine combines multiple SAS data into a combined consensus dataset   
e.g. data measured on different instruments on the same sample or SAXS and WAXS data    

## Download
sascombine is a python3 program, so you need python3    
download sascombine.py and the other python scripts in this repository       
see dependencies  

## Run  

#### standard run for combining 3 datasets
python sacombine.py --path example_data/ --data "dataset1.dat dataset2.dat dataset3.dat" --title my_combined_data

#### alternative command for combining the 3 datasets
python sacombine.py --path example_data --ext dat --title my_combined_data

#### instructions and options
python sascombine.py --help

## Dependencies

### dependencies from this folder:     
* sascombine_functions.py  

### other dependencies (standard python packages):   
* argparse     
* numpy    
* matplotlib    
* os    
* shutil    
* math    
* scipy
* sys
* time

these can usually be added to your python environment by installing pip and running, e.g., pip install numpy (or pip3 install numpy)    

## Notes  and warnings
* the program is a beta version - so use it with care. But feedback is more than welcome    
* sascombine can be applied to SANS data, but resolution effects are not taken into account, which will inevitably lead to systematic errors

## Credit
The program was written by Andreas Haahr Larsen   
Input and insight from Jochen Hub, Jill Trewhella and Patrice Vachette   
