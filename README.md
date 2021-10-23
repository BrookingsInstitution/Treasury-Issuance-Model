# MaturityStructure

## Abstract:
1. What the project does: This project contributes code to replicate and extend Belton et al (2018), Hutchins Center Working Paper #46
"Optimizing the maturity structure of U.S. Treasury debt: A model-based framework". The replication is still incomplete: the optimization module is still missing. The paper is extended by allowing the model to also consider Treasury Inflation Protected Securities (TIPS) and Floating Rate Notes (FRNs).   
2. Why it is useful: the model provides a tool to evaluate the tradeoff involved with the choice of different debt security issuance strategies.

## To get started:
Start from the files in doc folder: the paper Belton et al.pdf and the Jupyter notebook Treasury_Issuance_Model.pynb.

## Summary:
The project includes:
1. bin  folder: 
    1. Simulation.py : Python code, less human readable (but more compact) than Jupyer code. 
3. data folder:
    1. ACM.csv : data from Adrian, Crump, Moench (2013, updated). Source: https://www.newyorkfed.org/research/data_indicators/term_premia.html
    2. LW.csv  : data from Laubach Williams (2003, updated). Source: https://www.newyorkfed.org/research/policy/rstar
    3. FVALUESq.csv, COURATEq.csv : own re-elabotation derived from CRSP data. 
4. doc folder: 
    1. Treasury_Issuance_Model.pynb: Jupyter code, illustrating the code with explanations.  
    2. Belton et al.pdf : original paper. Also find it at https://www.brookings.edu/research/optimizing-the-maturity-structure-of-u-s-treasury-debt/

##  Contacs:
To report any mistakes, or get help with the code, please contact lrigon@stanford.edu. 

## Credits :
I, Lorenzo Rigon, wrote the code. All mistakes are mine.
The project was executed under the supervision of Brian Sack, with precious input from Zachary Harl and Terry Belton.
The project still has work in progress (add the simulation module of the paper, and add further extensions). 
