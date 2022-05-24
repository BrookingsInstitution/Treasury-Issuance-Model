# MaturityStructure

## Abstract:
1. What the project does: This project contributes code to replicate and extend Belton et al (2018), Hutchins Center Working Paper #46
"Optimizing the maturity structure of U.S. Treasury debt: A model-based framework". The replication is still incomplete: the dynamic optimization module is still missing. However, the paper is extended by allowing the model to also consider Treasury Inflation Protected Securities (TIPS) and Floating Rate Notes (FRNs).   
2. Why it is useful: the model provides a tool to evaluate the tradeoff involved with the choice of different debt security issuance strategies.

## To get started:
Start from the files in doc folder: the paper Belton et al.pdf and the Jupyter notebook Treasury_Issuance_Model.pynb.

## Summary:
The project includes:
1. bin  folder: 
    1. sim_lib.py : Python code, defines all the functions called in the Jupyter code (see below)
3. data folder:
    1. ACM.csv : data from Adrian, Crump, Moench (2013, updated). Source: https://www.newyorkfed.org/research/data_indicators/term_premia.html
    2. LW.csv  : data from Laubach Williams (2003, updated). Source: https://www.newyorkfed.org/research/policy/rstar
    3. FVALUESq.csv, COURATEq.csv : own re-elabotation derived from CRSP data. 
    4. MSPD_MktSecty_20010131_20211031 and MSPD_NonmktSecty_20010131_20211031: from MSPD dataset.
    5. FVmspd_q, CPRATEmspd_q: re-elaboration of MSPD dataset
4. doc folder: 
    1. Treasury_Issuance_Model-refactored.pynb: Jupyter code, illustrating the code with explanations.  
    2. Belton et al.pdf : original paper. Also find it at https://www.brookings.edu/research/optimizing-the-maturity-structure-of-u-s-treasury-debt/

##  Contacs:
To report any mistakes, or get help with the code, please contact lrigon@stanford.edu. 

## Credits :
I, Lorenzo Rigon, wrote the code. All mistakes are mine.
The project was executed under the supervision of Brian Sack, with precious input from Zachary Harl and Terry Belton.
The project still has work in progress (add the dynamic optimization, and add further extensions, in particular, SOMA). 
