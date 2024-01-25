# ML-SAScombine
version beta0.12

ML-SAScombine use a maximum likelihood (ML) approach to combines multiple small-angle scattering (SAS) datasets into a consensus dataset   
The input could be, e.g., data measured on different instruments on the same sample or SAXS and WAXS data (Trewhella, Vachette, Larsen 2024, in prep)   

## Download
ML-SAScombine is a python3 program, so you need python3    
Download mlsascombine.py and the other python scripts in this repository       
See dependencies  

## Run  

#### standard run for combining 3 datasets
python mlsascombine.py --path example_data/ --data "dataset1.dat dataset2.dat dataset3.dat" --title my_combined_data

#### alternative command for combining the 3 datasets
python mlsascombine.py --path example_data --ext dat --title my_combined_data

#### instructions and options
python mlsascombine.py --help

## Dependencies

### dependencies from this folder:     
* mlsascombine_functions.py  

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
* ML-SAScombine can be applied to SANS data, but resolution effects are not taken into account, which will inevitably lead to systematic errors

## Credit
The program was written by Andreas Haahr Larsen   
Input and insight from Jochen Hub, Jill Trewhella and Patrice Vachette   
