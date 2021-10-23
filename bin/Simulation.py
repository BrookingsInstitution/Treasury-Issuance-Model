from time import time
tic = time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler # For customized plot colors. 
import warnings; warnings.simplefilter('ignore', np.RankWarning) # Suppress warnings triggered by saturating degrees of freedom for fitting polynomials.
try:
    import cupy as cp  
    print('Cupy is installed, GPU will be used')
    xp = cp                 # Use Cupy if available in Cupy/Numpy agnostic functions. 
    pool = cp.get_default_memory_pool(); pool.free_all_blocks() # Not using Cupy unified/managed memory pool: seems to have negative effect on GPU memory and speed.
    TotDedicMem=cp.cuda.Device(device=cp.cuda.runtime.getDevice()).mem_info[0] #Total free dedicated memory found on default device (May be below total memory...)
except ImportError:
    print('Cupy is not installed, GPU will not be used')
    xp = np                 # If Cupy is not available, fall back to Numpy
try: 
    import os
    os.chdir('bin')  # Make sure your current directory is bin folder (necessary for relative path to data files). 
except:
    print("Current directory is already bin folder")

path_CRSP = None                             # Either this dataset or our small csv files COURATEq, FVALUESq derived from it must be present in your folder. 
ReplicateBelton = False                       # Set to true replicate Belton et al Fig 4, 5, 6, 11, 13 ...

#... Otherwise modify these settings below as desired (e.g. including TIPS and FRNs)... ###################################################################################################################################  
startyear = 2017
startquarter = 4
CBO_weight = 1                      # How much to adjust primary deficit mean towards CBO projection below. 
CBO_projection = xp.array([ -2.45]) # Vector up to length 10 with path of CBO-projected deficit for first 10 years after start date. Mean of deficits is adjusted towards this(these) number(s) for the first 10 years, then the effect tapers out by year 15. Please refer to Excel files for 10-Year Budget Projection at https://www.cbo.gov/data/budget-economic-data#3. 
plotFigs = True                     # Do plots for intermediate results (Initial debt profiles, details on rates)? 
No_TIPS_FRN = False                 # If True, TIPS and FRNs are not included in initial outstanding debt profiles
n_simula = 10000                     # In Cupy is used, Values above 10000 may create GPU memory issues on GPUs with 3.5gb of free space, with 80 periods of time.
n_period = 80                       # In quarters. 
n_exp_horizon = 201                 # Includes current quarter zero (just ended at end of period quarterly date) and 50y x 4 = 200 quarters forward.
use10y5yslope = True                # If False, extrapolates Term premia using 10y-2y slope (as in the paper) rather than the 10y-5y slope
use_convadj = True                  # If True, convexity adjustment is used to extrapolate ZCB Term Premia, which are then added to expectational component to derive ZCB rates, which are finally mapped to par rates. If False, convexity adjustment is not used to extrapolate term premia. 
replicateBeltonTP = True            # If True, a shift in par rates is implemented, accounting for the fact that par rates of Belton et al rely on a differet implementation of convexity adjustment. The shifts are set to match Belton et al par rates when other parameters are set for replication (by ReplicateBelton=True).
ELB = 0.125                         # Effective lower bound for FFR is 0.125% in Belton et al
Securities=np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 2 , 2 ],  # Securities (always Numpy array) must to have 2 rows: 1st row specifying security type (Nom=0, TIPS=1, FRN=2),
                     [1 , 2 , 3 , 5 , 7 , 10, 20, 30, 50, 2 , 5 , 10, 30, 2 , 5 ]]) # 2nd row specifying tenor, in years (from 0.25 to n_exp_horizon//4, always multiples of 0.25). Maximum is 50.   
Kernel1_Baseline = xp.reshape(xp.array([0.475,  0.11,  0.09, 0.115, 0.085,  0.08,   0.0, 0.045,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0], dtype= xp.float32), (-1,1)) # Baseline issuance shares. As many columns as Securities. Sum to 1. 
Kernel2_Bills    = xp.reshape(xp.array([ 1.00, -0.21, -0.17, -0.22, -0.16, -0.15,   0.0, -0.09,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0], dtype= xp.float32), (-1,1)) #Into Bills. Deviations from Baseline shares, sum to zero.
Kernel3_Belly    = xp.reshape(xp.array([-0.25,  0.25,  1.00,  0.50, -0.50, -0.75,   0.0, -0.25,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0], dtype= xp.float32), (-1,1)) #Into Belly. Deviations from Baseline shares, sum to zero.
Kernel4_Bonds    = xp.reshape(xp.array([ 0.00, -0.41, -0.33, -0.41, -0.10,  0.25,   0.0,  1.00,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0], dtype= xp.float32), (-1,1)) #Into Bonds. Deviations from Baseline shares, sum to zero.
QuartersperCoup = 2        # Select 1 for quarterly coupns, 2 for semiannual coupons. 
estimate_not_assume = True # Estimates Coefficients for ACM TP interpolation between 2y and 10y and then extrapolates w/conv adj out to 50y (rather than using assumed coeeficients out to 50y)

# Set model parameters as in Belton et al. 
L_ParNam =         ['rhoU1','betaUR','rhoU2','rhoZ1','betaPU','rhoP1','rhoP2','betaFTay','rhoF1','rhoEpsCPI','rhoEpsPRI','rhoEpsTP10','rhoEpsTP2','alphaZ','betaPPE2pct', 'sigmaU',  'sigmaG',  'sigmaZ' , 'sigmaP',  'sigmaNuCPI',  'sigmaNuPRI',  'sigmaNuTP10',  'sigmaNuTP2','ATP10', 'ATP2',    'alphaPRI', 'betaTP10U', 'BTP2U','betaTP10TP2','betaPRIU'] 
V_ParSrt = np.array([  1.57,  0.028 ,  -0.62,  0.917,  -0.133,   0.58,   0.26,      0.15,   0.85,      0.295,       0.92,        0.73,      0.63,(1-0.917)*(-0.5),0.16*2,     0.24,    0.0624,      0.018,     0.79,          1.70,          0.35,           0.41,          0.09,   0.51,  -0.05, 0.34,       0.207,-0.014+0.42*0.207,0.42,     -1.5 ])
#############################################################################################################################################
if ReplicateBelton== True:
    startyear = 2017
    startquarter = 4
    CBO_weight = 1                                 
    CBO_projection = xp.array([ -2.45])           # Belton et al primary deficit in initial periods is constantat at 2.45%. 
    plotFigs = True
    No_TIPS_FRN = False                            
    V_ParSrt[L_ParNam.index('alphaPRI')] -=  0.4  # Belton et al long term deficit does not settle to the constant alpha_PRI = 0.34 but to -0.2 (my non-zero long term UGAP induced by ELB is not enough to account for all of this difference, only pushes down from 0.34 to 0.2)
    n_simula = 10000
    n_period = 80
    n_exp_horizon = 201            
    use10y5yslope = False          
    use_convadj = True             # Notice that since we use conv. adjustments to extrapolate ZCB term premia (rather than TP on par rates whose duration is approximated before TP is added), our procedure and conv. adjustments are different than in Belton et al, so shifts to long end par rates will be needed for replication
    replicateBeltonTP = True       # Apply the shifts matching par rate curve. 
    ELB = 0.125                    # Belton et al has Effective lower bound for FFR at 0.125% 
    Securities=np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],  # Securities must to have 2 rows: 1st row specifying security type (Nom=0, TIPS=1, FRN=2),
                         [1 , 2 , 3 , 5 , 7 , 10, 20, 30, 50]]) # 2nd row specifying tenor, in years (from 0.25 to n_exp_horizon//4, always multiples of 0.25)  
    Kernel1_Baseline = xp.reshape(xp.array([0.475,  0.11,  0.09, 0.115, 0.08 , 0.085,   0.0, 0.045,    0.0], dtype= xp.float32), (-1,1))
    Kernel2_Bills    = xp.reshape(xp.array([ 1.00, -0.21, -0.17, -0.22, -0.16, -0.15,   0.0, -0.09,    0.0], dtype= xp.float32), (-1,1)) #Into Bills
    Kernel3_Belly    = xp.reshape(xp.array([-0.25,  0.25,  1.00,  0.50, -0.50, -0.75,   0.0, -0.25,    0.0], dtype= xp.float32), (-1,1)) #Into Belly
    Kernel4_Bonds    = xp.reshape(xp.array([ 0.00, -0.41, -0.33, -0.41, -0.10,  0.25,   0.0,  1.00,    0.0], dtype= xp.float32), (-1,1)) #Into Bonds
    QuartersperCoup = 1
    estimate_not_assume = True 

def F_InitStates(startyear,startquarter):
    import datetime as dtm
    startdate = dtm.datetime(startyear, int(startquarter*3), 30+1*(startquarter==1 or startquarter==4))
    from pandas_datareader.data import DataReader
    Q_series = ['NROU',                    # Natural Rate of Unemployment (Long-Term), Percent, Not Seasonally Adjusted
                'DPCCRV1Q225SBEA',         # Personal Consumption Expenditures (PCE) Excluding Food and Energy (chain-type price index), Percent Change from Preceding Period, Seasonally Adjusted Annual Rate
                'GDP',                     # Gross Domestic Product, Billions of Dollars, Seasonally Adjusted Annual Rate
                'UNRATE',                  # Unemployment Rate, Percent, Seasonally Adjusted 
                'FEDFUNDS'                 # Effective Federal Funds Rate, Percent, Not Seasonally Adjusted
                ] 
    Q_data = DataReader(Q_series, 'fred',startdate-dtm.timedelta(3*31), startdate) # Download data (start looking from start-of-quarter)
    startdate= str(int(startyear)) +'-'+ ('0' + str(int(startquarter*3)))[-2:] +'-'+ str(30+1*(startquarter==1 or startquarter==4))
    Q_data = Q_data.fillna(method='ffill').resample('Q').ffill() # Adjust dates convention: from Quarterly (or monthly) start-of-period to quarterly end-of-period
    Q_data = Q_data.loc[startdate]
    Init_GDP = int(Q_data['GDP']) # Nominal GDP, in current Billon USD
    V_StaSrt = np.zeros(14)
    V_StaSrt[[0,5]] = np.round(Q_data['UNRATE']-Q_data['NROU'],2) # Set initial UGAP. 
    V_StaSrt[[3,8]] = Q_data['DPCCRV1Q225SBEA']                   # Set initial PCE Inflation.
    V_StaSrt[[4,9]] = Q_data['FEDFUNDS']                          # Set initial Fed Funds Rate
    # Set initial Rstar from Laubach-Williams. 
    try:    # If LW dataset already downloaded in current folder, just read it
        LW = pd.read_csv('../data/LW.csv', index_col=0, parse_dates=True)
    except: # Otherwise download it and also save a copy to csv for future use (NOTE: may need updating for recent data)
        url_LW ="https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/Laubach_Williams_current_estimates.xlsx"
        LW = pd.read_excel(url_LW, 'data', index_col=0, parse_dates=True, header=5) # Download data
        LW  = LW.resample('Q').ffill()  # Resample dates to Quarterly end-of-period
        LW.to_csv('../data/LW.csv', index = True)
    Rstar = LW.loc[startdate,'rstar']     # Notice LW rstar = c g + z , with g potential growth, c a constant, and z autoregressive with mean zero. Belton et al has Rstar=Gstate+Zstate, with G potential growth and Zstate AR(1) with mean -0.5
    Gstate =LW.loc[startdate,'g']         # Take only potential growth g, not  c x g (c is the constant of Laubach Williams)
    Zstate =   Rstar - Gstate             # We define Z = r* - g , while LW have z = r* - cg, and z has mean zero, while Belton et al Z has mean Zss=0.5
    V_StaSrt[[1,6]] = Gstate                          # Set initial G Random Walk
    V_StaSrt[[2,7]] = Zstate                          # Set initial Z Autoregressive Process (Z has mean -0.5)
    V_StaSrt[[10,11]] = 0                             # Set initial EpsPRI and EpsCPI to zero. (Refinement may want to set EpsCPI to hit initial CPI)
    # Set initial EpsTP10, EpsTP2 to match initial TP10, TP2 from ACM. 
    try:    # If ACM dataset already downloaded in current folder, just read it
        ACM = pd.read_csv('../data/ACM.csv', index_col=0, parse_dates=True)
    except: # Otherwise download it and also save a copy to csv for future use
        url_ACM = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
        ACM = pd.read_excel(url_ACM, 'ACM Monthly', index_col=0, parse_dates=True) # Download data
        ACM  = ACM.resample('Q').ffill()  # Resample dates to Quarterly end-of-period
        ACM.to_csv('../data/ACM.csv', index = True)
    TP10 = ACM.loc[startdate,'ACMTP10'] 
    TP2  = ACM.loc[startdate,'ACMTP02']
    V_StaSrt[12] =   TP10  - V_ParSrt[L_ParNam.index('ATP10')] - V_ParSrt[L_ParNam.index('betaTP10U')]*V_StaSrt[0]
    V_StaSrt[13] =   TP2   - V_ParSrt[L_ParNam.index('ATP2')]  - V_ParSrt[L_ParNam.index('BTP2U')]    *V_StaSrt[0] - V_ParSrt[L_ParNam.index('betaTP10TP2')]*TP10
    return V_StaSrt, Init_GDP    

#Use CRSP Monthly Treasury Dataset to calibrate initial Debt and Average Coupon Rates Profile across tenors. 
#Notice that there are no FRNs in the dataset.
def F_InitiProfiles(startyear,startquarter, path_CRSP, plotFigs=True):
    try:    # If CRSP dadaset has already been prepared, just read results
        FVALUESq = pd.read_csv('../data/FVALUESq.csv', index_col=0, parse_dates=True)
        COURATEq = pd.read_csv('../data/COURATEq.csv', index_col=0, parse_dates=True)
    except: # Otherwise do the work to prepare the data and then save a copy for next use. 
        CRSP = pd.read_excel(path_CRSP, parse_dates=True) # Read CRSP data
        # Replace 'Non Answerable' instances of total Face Value inferring it from closest history of same security
        for nanobs in CRSP.index[CRSP['TMTOTOUT'].isna()]:
            temp = CRSP.loc[CRSP['KYTREASNO'] == CRSP.loc[nanobs,'KYTREASNO'], ['TMTOTOUT', 'MCALDT']] # Get history of security with a missing Face value entry
            if temp.index.size > 1:
                CRSP.loc[nanobs,'TMTOTOUT' ] = CRSP.loc[(temp.loc[temp.index != nanobs, 'MCALDT'] - temp.loc[nanobs,'MCALDT'] ).abs().idxmin(),'TMTOTOUT']    
        CRSP=CRSP[CRSP['TMTOTOUT'].isna()==False] #Drop the remaining nan observations (less than 50 obs out of around 150 000)
        CRSP['TMPRIOUT'] =  CRSP['TMTOTOUT'] - CRSP['TMPUBOUT'] # Privately held values are Total - Publicly Held.

        # Code to first build monthly horizons, in order to also have debt repaid in just finished quarter (HORIZON 0)
        CRSP['curMnt'] = CRSP['MCALDT'].dt.year*12+CRSP['MCALDT'].dt.month
        CRSP['endMnt'] = CRSP['TMATDT'].dt.year*12+CRSP['TMATDT'].dt.month
        CRSP['MNTSTOMAT'] = CRSP['endMnt']-CRSP['curMnt'] 

        # Compute outstanding face value profile, also distinguishing private and total outstanding
        CRSP['TMTOTOUT_ofTIPS'] = CRSP['TMTOTOUT'] * CRSP['ITYPE'].isin([11,12])
        CRSP['TMTOTOUT_noTIPS'] = CRSP['TMTOTOUT'] * (1-CRSP['ITYPE'].isin([11,12]))

        FVALUESm = CRSP[['MCALDT','MNTSTOMAT', 'TMTOTOUT_noTIPS', 'TMTOTOUT_ofTIPS']].fillna(np.inf).groupby(by=['MCALDT', 'MNTSTOMAT']).sum().replace(np.inf, np.nan)
        FVALUESm = FVALUESm.unstack('MNTSTOMAT', fill_value = 0).resample('M').ffill()                                                                # Horizon to maturity used to create columns. 
        FVALUESm[[('TMTOTOUT_noTIPS',0), ('TMTOTOUT_ofTIPS',0)]] = FVALUESm[[('TMTOTOUT_noTIPS',1), ('TMTOTOUT_ofTIPS',1)]].shift(1)                  # Assume FV repaid in month just ended was the outstanding for that month measured at the end of previous month.
        FVALUESq= FVALUESm.fillna(np.inf).groupby(lambda x: 'FV' + x[0][9:15] + str(np.int(np.ceil(x[1]/3))) , axis=1).sum().replace(np.inf, np.nan)  # Aggregate horizons to quarters. Note how 0th columns are unchanged.
        FVALUESq[['FVofTIPS0', 'FVnoTIPS0']] = FVALUESq[['FVofTIPS0', 'FVnoTIPS0']].rolling(3).sum(skipna=False);                                     # Aggregate FV repaid in last 3 months to get FV repaid in last quarter. 
        FVALUESq = FVALUESq.resample('Q').ffill() 
        FVALUESq.to_csv('FVALUESq.csv', index = True)

        # Compute weighted average coupon rate profile
        COURATEm = CRSP[['MCALDT','MNTSTOMAT','TMTOTOUT_noTIPS','TMTOTOUT_ofTIPS','TCOUPRT']] #'TMYLD' has strictly worse coverage than 'TMPCYLD', and substantially same info on YTM.
        COURATEm = COURATEm.assign(ANNCOUSIZE_noTIPS = COURATEm['TCOUPRT']*0.01*COURATEm['TMTOTOUT_noTIPS'], ANNCOUSIZE_ofTIPS = COURATEm['TCOUPRT']*0.01*COURATEm['TMTOTOUT_ofTIPS'], QTRSTOMAT = np.ceil(COURATEm['MNTSTOMAT']/3)).drop(columns=['TCOUPRT', 'MNTSTOMAT'] )
        COURATEm = COURATEm.fillna(np.inf).groupby(by=['MCALDT', 'QTRSTOMAT']).sum().replace(np.inf, np.nan) 
        COURATEm = COURATEm.assign(AVGCOURT_noTIPS = 100*COURATEm['ANNCOUSIZE_noTIPS']/COURATEm['TMTOTOUT_noTIPS'], AVGCOURT_ofTIPS = 100*COURATEm['ANNCOUSIZE_ofTIPS']/COURATEm['TMTOTOUT_ofTIPS']).drop(columns=['TMTOTOUT_noTIPS','ANNCOUSIZE_noTIPS','TMTOTOUT_ofTIPS','ANNCOUSIZE_ofTIPS'])
        COURATEq = COURATEm.unstack('QTRSTOMAT', fill_value = np.nan).resample('Q').ffill()                 # Horizon to maturity used to create columns. 
        COURATEq = COURATEq.fillna(np.inf).groupby(lambda x: x[0] + str(np.int(x[1])) , axis=1).sum().replace(np.inf, np.nan) #Consolidate 2 levels of labels to column titles
        COURATEq['AVGCOURT_noTIPS0'] = np.NaN; COURATEq['AVGCOURT_ofTIPS0'] = np.NaN
        COURATEq.to_csv('COURATEq.csv', index = True)
    # Use Data to set initial debt and coupon profiles. 
    startdate= str(int(startyear)) +'-'+ ('0' + str(int(startquarter*3)))[-2:] +'-'+ str(30+1*(startquarter==1 or startquarter==4))
    MaxAhead = int(FVALUESq.shape[1]/2); 
    Init_DbtFVout=np.zeros((n_exp_horizon,2));  
    Init_DbtFVout[0:min(MaxAhead,n_exp_horizon),0] = FVALUESq.loc[startdate, ['FVnoTIPS' + str(x) for x in np.arange(MaxAhead)]]
    Init_DbtFVout[0:min(MaxAhead,n_exp_horizon),1] = FVALUESq.loc[startdate, ['FVofTIPS' + str(x) for x in np.arange(MaxAhead)]]
    Init_DbtFVout /= 1000 ;  #Divide by 1000 to change units from millions to billions.  
    MaxAheadc = int(COURATEq.shape[1]/2)
    Init_AvgCoupRate=np.empty((n_exp_horizon,2)); Init_AvgCoupRate[:]= np.nan; 
    Init_AvgCoupRate[0:min(MaxAheadc,n_exp_horizon),0] = COURATEq.loc[startdate, ['AVGCOURT_noTIPS' + str(x) for x in np.arange(MaxAheadc)]]
    Init_AvgCoupRate[0:min(MaxAheadc,n_exp_horizon),1] = COURATEq.loc[startdate, ['AVGCOURT_ofTIPS' + str(x) for x in np.arange(MaxAheadc)]]
    # Plot the initial profiles
    if plotFigs==True:
        Init_AvgCoupRate_toplot = np.copy(Init_AvgCoupRate); 
        Init_AvgCoupRate_toplot[np.isnan(Init_AvgCoupRate_toplot)]=-99
        Ticks= [str(round(x/4)) + 'y' for x in np.arange(0,n_exp_horizon,20)] 
        fig, ax = plt.subplots(2,2, sharex=True)
        for col in range(2):
            ax[0,col].plot(Init_DbtFVout[:,col], c='red'); ax[0,col].set_ylabel('Bn USD'); ax[0,col].set_yscale('linear'); ax[0,col].legend(['Face Value Outstanding by tenor']); ax[0,col].set_title('Initial '+ ['Nominal', 'TIPS'][col] +' Debt Profile at start date ' + startdate)
            ax[1,col].plot(Init_AvgCoupRate_toplot[:,col], c='black'); 
            ax1r = ax[1,col].twinx(); ax1r.plot(Init_DbtFVout[:,col], '--',c='red',linewidth=0.75); ax1r.set_ylabel('Bn USD'); ax1r.set_yscale('log'); ax1r.legend(['Log Face Value Outstanding'], loc='lower right'); 
            ax[1,col].set_xticks([x for x in np.arange(0,n_exp_horizon,20)]); ax[1,col].set_xticklabels(Ticks); ax[1,col].set_ylabel('%'); ax[1,col].legend(['Average Coupon Rate on ' + ['Nominal', 'TIPS'][col] + ' Debt by tenor']); ax[1,col].set_title('Initial Average ' + ['Coupon', 'TIPS'][col] + ' Rate Profile at start date ' + startdate)
            ax[1,col].set_xlim((0,4*40)); ax[1,col].set_ylim(0, max(Init_AvgCoupRate_toplot[:,col])+1); ax[0,col].set_ylim(0); ax1r.set_ylim(1,max(Init_DbtFVout[:,col])*50 ); ax[1,col].set_xlabel('Tenor: years ahead')
    Init_TipsFVadj = Init_DbtFVout[:,1]; Init_FrnsFV =   xp.zeros(n_exp_horizon) # Temporary, till we retrieve data on CPI index ratios and FRNs securities from MSPD
    if No_TIPS_FRN== True:
        Init_DbtFVout[:,1]*=0; Init_AvgCoupRate[:,1]*=0; Init_TipsFVadj*=0; Init_FrnsFV*=0 #Drop TIPS and FRNs initial securities.
    return xp.asarray(Init_DbtFVout, dtype=xp.float32), xp.asarray(Init_AvgCoupRate, dtype=xp.float32), xp.asarray(Init_TipsFVadj, dtype=xp.float32), xp.asarray(Init_FrnsFV, dtype=xp.float32)

# Initialize matrices and update parameters for linear version of Belton et al (ZLB will be tackled with extra code in transition equation)
# This function to create the model matrices looks like REALLY ugly code but on average is twice as fast (and easier to visually inspect) than plugging parameters in indexed positions as done in another versions of code. 
def F_BeltonMatrices(p):
    if xp != np:
        if cp.get_array_module(p) == cp: 
            p = p.get() # Make sure params are Numpy

    M_Transi=xp.array([[  p[0] ,-p[1]/2,-p[1]/2,-p[1]/2, p[1]/2,   p[2],-p[1]/2,-p[1]/2,-p[1]/2, p[1]/2,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,  p[3] ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [  p[4] ,   0.0 ,   0.0 ,   p[5],   0.0 ,   0.0 ,   0.0 ,   0.0 ,  p[6] ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [-2*p[7],  p[7] ,  p[7] ,1.5*p[7], p[8] ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   p[9],   0.0 ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,  p[10] ,   0.0 ,   0.0],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,  p[11],   0.0],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 , p[12]]], dtype= xp.float32)

    V_ConSta=xp.array( [   0.0 ,   0.0 ,  p[13],  p[14],  -p[7],   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ], dtype= xp.float32)

    V_ConObs=xp.array( [   0.0 ,   0.0 ,   0.0 ,   0.0 ,  p[23], p[24], p[25]], dtype= xp.float32)

    M_Design=xp.array([[   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ,   0.0 ],
                       [  p[26],   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ],
                       [  p[27],   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,  p[28],   1   ],
                       [  p[29],   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   1   ,   0.0 ,   0.0 ]], dtype= xp.float32)

    M_CovSho=xp.array([[p[15]**2,  0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,p[16]**2,  0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,p[17]**2,  0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,p[18]**2,  0.0 ,   0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,p[19]**2,  0.0 ,   0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,p[20]**2,  0.0 ,   0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,p[21]**2,  0.0 ],
                       [   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,p[22]**2]], dtype= xp.float32)

    M_Select=xp.array([[   1  ,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   1  ,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   1  ,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   1  ,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   1  ,   0.0,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   1  ,   0.0,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1  ,   0.0 ],
                       [   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1   ]], dtype= xp.float32)    
    V_Stdvs = M_Select @ xp.diag(M_CovSho)**0.5
    return locals().copy() 
ModelMats = F_BeltonMatrices(V_ParSrt)

def F_SimSta(n_period, n_simula, V_StaSrt, ModelMats): # Computes just macro block in case 10 rather than 13 starting states are passed. 
    states = len(V_StaSrt)
    A_Stdevs = xp.swapaxes(xp.array(ModelMats['V_Stdvs'], ndmin=3), 2, 1)
    M_Transi = ModelMats['M_Transi'] 
    A_ConSta = xp.swapaxes(xp.tile(ModelMats['V_ConSta'], (n_simula,1)),0,1)
    A_SimShock2States = xp.zeros((n_period,states,n_simula), dtype= xp.float32)
    if xp != np: A_SimShock2States  = xp.random.standard_normal(size=(n_period,states,n_simula),dtype = xp.float32) # Important for GPU memory to use directly float32 rather than float64 and then convert. Directly create 13 columns (rather than 7) since this object will also store states, and expanding an array can't be done in place. 
    else:        A_SimShock2States  = xp.random.standard_normal(size=(n_period,states,n_simula)).astype(xp.float32) # Numpy does not support dtype argument in normal creation
    A_SimShock2States *= A_Stdevs                                                     # Multiply in place to reduce memory use. Scaling by Stdevs is more efficient than drawing from multivariate normal, or matmult by VCov
    A_SimShock2States[0,:,:] = xp.swapaxes(xp.tile(V_StaSrt, (n_simula,1)),0,1)       # Plug in initial state vector 
    for t in range(1,n_period):                                                       # Shocks are used to compute states, then immediately overwritten with computed states to save memory.
        A_SimShock2States[t,:,:] += A_ConSta + M_Transi @ A_SimShock2States[t-1,:,:]  # Transition Equation w/out ZLB
        #A_SimShock2States[t,4,:] *= (A_SimShock2States[t,4,:]>0)                      # Zero Lower Bound.
        xp.clip(A_SimShock2States[t,4,:], ELB, None, out = A_SimShock2States[t,4,:]) # Effective Lower Bound at 0.125 %
    return A_SimShock2States
# Compile function on small arrays
V_StaSrt = np.array([-0.5,1.5,-1,1.8, 2   ,-0.5,1.5,-1,1.8 ,2   ,0,0,-0.25,0]) # Initial states, eyeballed from Belton et al Fig 4, and zero initial AR1 errors except for EpsTP10 at -0.25.
A_SimSta = F_SimSta(n_period, 5000, V_StaSrt , ModelMats); del A_SimSta  

# Most of GPU time is used to bring GPU arrays back to CPU (need to allocate CPU memory), or outside this function to delete GPU arrays from previous run (in order to free GPU memory). 
def F_SimObs(A_SimSta, ModelMats):
    n_period, n_simula  = A_SimSta.shape[0], A_SimSta.shape[2]
    M_Design = ModelMats['M_Design']
    A_SimObs = xp.empty((n_period,M_Design.shape[0],n_simula), dtype=xp.float32)
    A_ConObs = xp.swapaxes(xp.tile(ModelMats['V_ConObs'], (n_simula,1)),0,1)
    for t in range(0,n_period): # Shocks are used to compute states, then immediately overwritten with computed states to save memory.
        A_SimObs[t,:,:] = A_ConObs + M_Design @ A_SimSta[t,:,:] # Observation equation: would work a little faster outside loop with "A_SimObs = A_ConObs + (ModelMats['M_Design'] @ A_SimShock2States)",  but uses more memory, which is critical with GPU. 
        CBOaddon =  (CBO_projection[min(t//4, len(CBO_projection)-1)]*(1-(t%4)/4) + CBO_projection[min(1+t//4, len(CBO_projection)-1)]*(t%4)/4 ) - xp.mean(A_SimObs[t,6,:], axis=0, keepdims=True)
        A_SimObs[t,6,:] +=  CBO_weight*CBOaddon  * ((t<41) + (t>40)*(t<61)*(1-(t-40)/20))
    return A_SimObs
# Compile function on small arrays
A_SimObs = F_SimObs(F_SimSta(n_period, 5000, V_StaSrt , ModelMats), ModelMats); del A_SimObs

def MakeFFPaths3(A_SimSta, ModelMats, A_FFPaths, A_CPIPaths, n_exp_horizon=201): 
    A_SimStaCut = xp.copy(A_SimSta[:,0:11,:])                   # Cut out PRI and TP AR1 states (if passed) to focus on Macro Block core eleven states. Make local copy of to keep overwriting it in loop
    M_Transi = ModelMats['M_Transi'][0:11,0:11]                 # Focus on Macro Block core states 
    A_ConSta = xp.swapaxes(xp.tile(ModelMats['V_ConSta'][0:11], (A_SimStaCut.shape[0],A_SimStaCut.shape[2],1)),1,2) #Pre-bradcast to shape of A_SimStaCut
    A_FFPaths[:,0,:] = A_SimStaCut[:,4,:]
    A_CPIPaths[:,0,:] = A_SimStaCut[:,3,:]+A_SimStaCut[:,10,:]
    for t in range(1,n_exp_horizon):                                    # Shocks are used to compute states, then immediately overwritten with computed states to save memory.
        A_SimStaCut = xp.matmul(M_Transi, A_SimStaCut, out=A_SimStaCut) # Transition Equation 1) Matrix multiplication in place. 
        A_SimStaCut += A_ConSta                                                        #2) Addition in place
        #A_SimStaCut[:,4,:] *= (A_SimStaCut[:,4,:]>0)                                  #3) Zero Lower Bound. 
        xp.clip(A_SimStaCut[:,4,:], ELB, None, out = A_SimStaCut[:,4,:])               #3) Effective Lower Bound at 12.5 bps annualized rate
        A_FFPaths[:,t,:] = (A_FFPaths[:,t-1,:]*(t-1) + A_SimStaCut[:,4,:])/t     # Directly compute average FFrate till horizon
        if t < 30*4+1: #Keep CPI array smaller (5y and 10y tenors only needed for TIPS's IRP, tenors out to 30y needed for FRNs)
            A_CPIPaths[:,t,:]=(A_CPIPaths[:,t-1,:]*(t-1) + A_SimStaCut[:,3,:]+A_SimStaCut[:,10,:])/t   # Same for average CPI
# Compile on small run
A_SimSta = F_SimSta(8, 2, V_StaSrt, ModelMats); A_FFPaths= xp.empty(( A_SimSta.shape[0],n_exp_horizon,A_SimSta.shape[2] ), dtype=xp.float32); 
A_CPIPaths= xp.empty(( A_SimSta.shape[0],30*4+1,A_SimSta.shape[2] ), dtype=xp.float32)
MakeFFPaths3(F_SimSta(8, 2, V_StaSrt, ModelMats), ModelMats, A_FFPaths, A_CPIPaths);  del A_FFPaths, A_SimSta, A_CPIPaths

# Function to create interpolated coefficients of ACM term premia at all quarters of tenor between tenor = 2y and tenor = 10y, included,
# against constant, ACMTP2y, ACMTP10y
def MakeCoeffTP2_TP10(plot_coeff = True):
    try:    # If ACM dataset already downloaded in current folder, just read it
        ACM = pd.read_csv('../data/ACM.csv', index_col=0, parse_dates=True)
    except: # Otherwise download it and also save a copy to csv for future use
        url_ACM = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
        ACM = pd.read_excel(url_ACM, 'ACM Monthly', index_col=0, parse_dates=True) # Download data
        ACM  = ACM.resample('Q').ffill()  # Resample dates to Quarterly end-of-period
        ACM.to_csv('../data/ACM.csv', index = True)
    # Run linear regressions of ACM Term premia at 3,4,5,6,7,8,9 years tenors against intecept and 2 and 10 years tenors. 
    import statsmodels.formula.api as smf
    Reg_Coeff = {}; Dep_Variables = ['ACMTP0' + str(x) for x in range(3,10)]
    for dep_var in Dep_Variables :
        Reg_Coeff[dep_var] = smf.ols(dep_var + '~ ACMTP02 + ACMTP10', data=ACM).fit().params # Store linear regression results (extract coefficients only)
    # Interpolate coefficients for all quarters between 2 and 10 years with n-degree polynomial (order n chosen to exhaust degrees of freedom)
    Tenor_y = np.arange(2,11)
    order =  len(Tenor_y)
    Tenor_q = np.arange(2,10.25,0.25)
    Coeff_q = {}; Coeff_y = {} 
    for Regressor in ['Intercept', 'ACMTP02', 'ACMTP10']:
        Coeff_y[Regressor] = np.array([1*(Regressor=='ACMTP02')] + [Reg_Coeff[dep_var][Regressor] for dep_var in Dep_Variables] + [1*(Regressor=='ACMTP10')]) # Unpack estimated coefficients of ACMTP3y,4y,...9y against Intecept, Beta to ACMTP2y, Beta to ACMTP10y and append the coefficients of ACMTP2, ACMTP10 themselves (0,1,0 and 0,0,1).
        Eval_Coeff = np.poly1d(np.polyfit(Tenor_y, Coeff_y[Regressor] , order))        # Fit polynomyal toy yearly coefficients, then build evaluator for polynomial
        Coeff_q[Regressor] = Eval_Coeff(Tenor_q )                               #Evaluate Polynomial
    if plot_coeff == True:
        fig0 = plt.figure(figsize=[16/1.1, 9/1.1]); plt.plot( Tenor_q , Coeff_q['Intercept'], '.', Tenor_y, Coeff_y['Intercept'], '*', Tenor_q , Coeff_q['ACMTP02'], '.', Tenor_y, Coeff_y['ACMTP02'], '*', Tenor_q , Coeff_q['ACMTP10'], '.', Tenor_y, Coeff_y['ACMTP10'], '*')
        plt.legend(('Interpolated Intercept', 'Estimated Intercept', 'Interpolated Beta to ACMTP2', 'Estimated Beta to ACMTP2', 'Interpolated Beta to ACMTP10', 'Estimated Beta to ACMTP10'), loc='best')
        plt.xlabel('ACM Term premium at tenor (years)'); plt.ylabel('OLS Coefficient value'); plt.title('Regression of ACM Term premia on ACM TP10, ACM TP2');  #plt.show(block=False);  
    return Coeff_q

def AssumeCoeffTP2_TP10(plot_coeff=True):
    Tenor_y = np.array([2,3,5,7,10,20,30,50])
    Tenor_q = np.arange(2,50.25,0.25)
    Coeff_q = {}; Coeff_y = {}
    Coeff_y['Intercept'] = np.array([0,     0,     0,     0,    0,    0.5, 0.4064, 0.45])
    Coeff_y['ACMTP02']   = np.array([1, 0.847, 0.514, 0.255,    0,-0.0397,-0.6203, -.65])
    Coeff_y['ACMTP10']   = np.array([0, 0.185, 0.506, 0.746,    1, 1.0234, 1.2366, 1.3])
    for Regressor in ['Intercept', 'ACMTP02', 'ACMTP10']:
        #Coeff_y[Regressor] = np.append(Coeff_y[Regressor], np.poly1d(np.polyfit(Tenor_y[:-1], Coeff_y[Regressor] ,1))(50) )  # Extrapolate to add 50y point to Coeff_y
        Coeff_q[Regressor] = np.interp(Tenor_q, Tenor_y, Coeff_y[Regressor])  # Interpolate to get Coeff_q
    if plot_coeff == True:
        fig0 = plt.figure(figsize=[16/1.1, 9/1.1]); plt.plot( Tenor_q , Coeff_q['Intercept'], '.', Tenor_y, Coeff_y['Intercept'], '*', Tenor_q , Coeff_q['ACMTP02'], '.', Tenor_y, Coeff_y['ACMTP02'], '*', Tenor_q , Coeff_q['ACMTP10'], '.', Tenor_y, Coeff_y['ACMTP10'], '*')
        plt.legend(('Interpolated Intercept', 'Assumed Intercept', 'Interpolated Beta to ACMTP2', 'Assumed Beta to ACMTP2', 'Interpolated Beta to ACMTP10', 'Assumed Beta to ACMTP10'), loc='best')
        plt.xlabel('ACM Term premium at tenor (years)'); plt.ylabel('OLS Coefficient value'); plt.title('Projection of ACM Term premia on ACM TP10, ACM TP2');  #plt.show(block=False); 
    return Coeff_q

def F_MakeIRP(A_ExpFFR_05_10, A_SimSta, A_CPIPaths, plot_IRPcoeff=True, plot_IRP = True):  
    Store_IRP=xp.zeros((A_SimSta.shape[0],1+4*30, A_SimSta.shape[2] ), dtype=xp.float32) 
    # Create AR1 residuals. Start by creating innovations
    if xp != np: IRP5_and_10 = xp.random.standard_normal(size=(A_SimSta.shape[0],2,A_SimSta.shape[2]),dtype = xp.float32) # Utterly important for GPU memory to use directly float32 rather than float64 and then convert. 
    else:        IRP5_and_10 = xp.random.standard_normal(size=(A_SimSta.shape[0],2,A_SimSta.shape[2])).astype(xp.float32) # Numpy does not support dtype argument in normal creation
    IRP5_and_10 *= 0.25 #Volatility of innovations to IRP5, IRP10 
    for t in range(1,A_SimSta.shape[0]): #Add up the innovations into the AR1 processes
        IRP5_and_10[t,:,:] += 0.7 * IRP5_and_10[t-1,:,:]  
    IRP5_and_10[:,0,:] += 0.61 - 0.145*((A_ExpFFR_05_10[:,0,:] - A_CPIPaths[:,5*4,:])  - (A_SimSta[:,1,:] + A_SimSta[:,2,:]))  # Add to AR1 to get IRP5
    IRP5_and_10[:,1,:] += 0.61 - 0.245*((A_ExpFFR_05_10[:,1,:] - A_CPIPaths[:,10*4,:]) - (A_SimSta[:,1,:] + A_SimSta[:,2,:]))  # Add to AR1 to get IRP10 
    Tenor_y = np.array([    2,    3,    5,    7,   10,   15,   20,   30])
    Consta  = np.array([    0,    0,    0,    0,    0,    0,    0,    0])
    Betas05 = np.array([ 1.02, 1.18,    1, 0.48,    0, 0.06, 0.10, 0.23])
    Betas10 = np.array([-0.49,-0.49,    0, 0.62,    1, 0.87, 0.88, 0.87])
    #Tenor_y = np.array([      2,      3,    5,      7,   10,     20,     30])
    #Consta  = np.array([   0.08,   0.06,    0,  -0.02,    0, -0.025,  -0.05])
    #Betas05 = np.array([ 1.4833, 1.3781,    1, 0.538 ,    0,-0.3569,-0.3076])
    #Betas10 = np.array([-0.6137,-0.4722,    0, 0.5038,    1, 1.2343, 1.1907])
    Tenor_q = np.arange(0,30.25,0.25)
    Consta_q  = np.interp(Tenor_q, Tenor_y, Consta)
    Betas05_q = np.interp(Tenor_q, Tenor_y, Betas05) 
    Betas10_q = np.interp(Tenor_q, Tenor_y, Betas10) 
    if plot_IRPcoeff==True: fig5 = plt.figure(figsize=[16/1.1, 9/1.1]); plt.plot(Tenor_q , Consta_q, '.', Tenor_y, Consta, '*', Tenor_q , Betas05_q, '.', Tenor_y, Betas05, '*', Tenor_q , Betas10_q, '.', Tenor_y, Betas10, '*'); plt.legend(('Interpolated Constants','Given Constants','Interpolated Betas to IRP 5y','Given Betas to IRP 5y','Interpolated Betas to IRP 10y','Given Betas to IRP 10y')); plt.title('Inflation Risk Premia Curve from 5y and 10y points')
    Store_IRP += xp.reshape(xp.asarray(Consta_q), (1,-1,1)) 
    Store_IRP += xp.atleast_2d(xp.asarray(Betas05_q)).T @ xp.expand_dims(IRP5_and_10[:,0,:],1)  
    Store_IRP += xp.atleast_2d(xp.asarray(Betas10_q)).T @ xp.expand_dims(IRP5_and_10[:,1,:],1)
    if plot_IRP == True: 
        fig6, axes = plt.subplots(nrows= 3, ncols= 2 , sharex=True, sharey='none', figsize=(16/2,9/2)); 
        if xp != np: Arrays = [xp.mean(Store_IRP[:,[x*4 for x in [2,3,5,7,10,15,20,30]],:],2).get(), xp.mean(A_ExpFFR_05_10,2).get(), xp.mean(A_SimSta[:,1,:] + A_SimSta[:,2,:],1).get(), xp.mean(A_CPIPaths[:,[4*5,4*10],:],2).get()]
        else:        Arrays = [xp.mean(Store_IRP[:,[x*4 for x in [2,3,5,7,10,15,20,30]],:],2)      , xp.mean(A_ExpFFR_05_10,2)      , xp.mean(A_SimSta[:,1,:] + A_SimSta[:,2,:],1)      , xp.mean(A_CPIPaths[:,[4*5,4*10],:],2)      ]
        axes[0, 0].set_prop_cycle(cycler(color=[plt.cm.get_cmap('rainbow')(x/len(Tenor_y)) for x in range(1,len(Tenor_y)+1)]))
        Legendpieces = [ str(Tenor_y[x])+'y = '+ str(round(Betas05[x],2))+ ' x 5y ' + ('+'+str(round(Betas10[x],2)))[-5:] + ' x 10y + eps_'+str(Tenor_y[x]) for x in range(len(Tenor_y))]; Legendpieces[2]='5y = 0.61 - 0.145 Rgap5'; Legendpieces[4]='10y = 0.61 - 0.245 Rgap10'
        lines = axes[0,0].plot(Arrays[0]); axes[0,0].set_title('Inflation Risk Premia (mean)'); axes[0,0].legend(iter(lines), Legendpieces, loc='lower right') 
        lines = axes[0,1].plot((Arrays[1]-Arrays[3])-np.expand_dims(Arrays[2],1)); axes[0,1].set_title('R gap = R - Rstar'); axes[0,1].legend(iter(lines), ('5y Rgap, mean', '10y Rgap, mean'), loc='upper right') 
        lines = axes[1,0].plot(Arrays[1]-Arrays[3]); axes[1,0].set_title('R = Exp Nom - Exp Inf'); axes[1,0].legend(iter(lines), ('5y R, mean', '10y R, mean'), loc='upper right') 
        lines = axes[1,1].plot(Arrays[2], 'k'); axes[1,1].set_title('R star (mean)'); axes[1,1].legend(('Rstar = Z state + G state'), loc='best')
        lines = axes[2,0].plot(Arrays[3]); axes[2,0].set_title('Expected inflation rate (avg future CPI, mean)'); axes[2,0].legend(iter(lines), ('5y horizon', '10y horizon'), loc='upper right') 
        lines = axes[2,1].plot(Arrays[1]); axes[2,1].set_title('Expected nominal rate (avg future FFR, mean)'); axes[2,1].legend(iter(lines), ('5y horizon', '10y horizon'), loc='upper right') 
    return Store_IRP

def MakeTPPaths2(A_SimObs, A_Storage, plot_coeff = False, plot_conv = False,  use10y5yslope=True, use_convadj=True, estimate_not_assume=True, TP_is_for_ZeroCurve = True, replicateBeltonTP=True):
    if estimate_not_assume == True: CoeffTP2_TP10 =   MakeCoeffTP2_TP10(plot_coeff = plot_coeff) # Time consuming unless ACM data already downloaded. Coeffs between 2y and 10y point.
    else:                           CoeffTP2_TP10 = AssumeCoeffTP2_TP10(plot_coeff = plot_coeff) # For checking results conditional on given coefficients, coeffs out to 50y point. 
    TP02series_q = xp.expand_dims(A_SimObs[:,5,:], 1)
    TP10series_q = xp.expand_dims(A_SimObs[:,4,:], 1)    
    Extrap_TP0_TP2_2(A_Storage, TP02series_q)
    TP05series_q = Interp_TP2_TP10_2(A_Storage, TP02series_q, TP10series_q,  CoeffTP2_TP10)
    if estimate_not_assume == True: Extrap_TP10_TP50_2(A_Storage, TP02series_q, TP05series_q, TP10series_q, plot_conv = plot_conv, use10y5yslope=use10y5yslope, use_convadj=use_convadj, replicateBeltonTP=replicateBeltonTP)
    if TP_is_for_ZeroCurve == True: MakeCoupRates(A_Storage) # Note if estimate_not_assume = True, TP_is_for_ZeroCurve should also be set to True
    A_Storage[:,0:5,:] -= 0.08 # Adjust for Bill-FFR basis, 8bps in Belton et al.
    MakeOnTheRun(A_Storage)    # Adjust for on-the run / off-the-run 

def Extrap_TP0_TP2_2(A_Storage,TP02series_q):
    for qtr in [0,1,2,3,4,5,6,7]: #Add term premium for first eight quarters: from +0qtrs (now), to +7qtr (1yr 2qtrs).
        #A_Storage[:,qtr,:] += xp.squeeze(TP02series_q) * qtr/4 * (qtr>4)  - (qtr<5)*0.08 # TP linearly increases in qtrs +4,...,+7 from 0 towards TP2(value at +8 qtrs) 
        A_Storage[:,qtr,:] += xp.squeeze(TP02series_q) * qtr/8 

def Interp_TP2_TP10_2(A_Storage, TP02series_q, TP10series_q, Coeff_q):
    cut = len(Coeff_q['Intercept'])+8 #Cutoff will be 41 (10y point) with estimation and 201 (50y) with assumption
    Intercept_q = xp.reshape(xp.asarray(Coeff_q['Intercept'], dtype=xp.float32), (1,-1,1))
    Beta_TP02_q = xp.reshape(xp.asarray(Coeff_q['ACMTP02'],   dtype=xp.float32), (-1,1))
    Beta_TP10_q = xp.reshape(xp.asarray(Coeff_q['ACMTP10'],   dtype=xp.float32), (-1,1))
    A_Storage[:,8:cut,:] += Intercept_q
    A_Storage[:,8:cut,:] += Beta_TP02_q @ TP02series_q  # Add is in place but Matmul is not... suboptimal for memory management.. but improving seems hard. Can divide in blocks if too big. 
    A_Storage[:,8:cut,:] += Beta_TP10_q @ TP10series_q # Extra: save separately TP05series_q as intermediate result for next function
    TP05series_q = Intercept_q[:,4*(5-2),:] + Beta_TP02_q[12] *  TP02series_q + Beta_TP10_q[12] *  TP10series_q
    return TP05series_q

#Transforms (in place) rates of Zero Coupon Bond into rates of coupon bonds issued at par (these rates are equal to their coupon rates)
def MakeCoupRates(A_IRPaths):
    DiscFactors = (1/(1+A_IRPaths/400))**xp.reshape(xp.arange(0,A_IRPaths.shape[1],dtype=xp.float32), tuple([1,-1] + [1 for x in  range(A_IRPaths.ndim-2)]) )
    #DiscFactors = xp.exp((-A_IRPaths/400)*xp.reshape(xp.arange(0,A_IRPaths.shape[2],dtype=xp.float32), tuple([1,1,-1] + [1 for x in  range(A_IRPaths.ndim-3)]) ))
    for T in range(1,A_IRPaths.shape[1]):
        A_IRPaths[:,T,...] = 400*(1-DiscFactors[:,T,...])/xp.sum(DiscFactors[:,1:T+1,...], axis=1)  

def Extrap_TP10_TP50_2(A_Storage, TP02series_q, TP05series_q, TP10series_q, plot_conv = False, use10y5yslope=True, use_convadj=False, replicateBeltonTP = True):
    Tenor_y =     np.array([    5,    10,    20,    30,    50])
    #ConAdj_ypar = np.array([1/100,10/100,30/100,51/100,81/100, 140/100, 247/100], dtype=np.float32) 
    #Tenor_ypar =  np.array([4.65 ,   8.5,  14.2,    18,    20,      30,      50])
    ConAdj_y = np.array([2.7  ,9.9  ,33.4 ,67.9 ,188.6], dtype=np.float32)/100  #These are adjustments for ZCBs are different from Belton et al. where convexity adjustments were though of as for par rates, whose duration for extrapolation was approximated by the duration without term premium. 
    #ConAdj_y = np.interp(Tenor_y, Tenor_ypar, ConAdj_ypar)
    #fig1 = plt.figure(figsize=[16/1.1, 9/1.1]); plt.plot(Tenor_y, 100*ConAdj_yInf, '.', Tenor_ypar , 100*ConAdj_ypar, '*', Tenor_y , 100*ConAdj_y, 'v'); plt.legend(('Interpolation at 5y,10y', 'Paper Conv. Adj. for par mapped to ZCB duration + my assumptions for 30y, 50y', 'Brian New Conv. Adj for ZCBs' ), loc='best'); plt.xlabel('Conv. Adjustment at duration (years)'); plt.ylabel('basis points'); plt.title('Convexity Adjustments')
    Tenor_q = np.reshape(np.arange(0,50.25,0.25, dtype=np.float32), (-1,1))
    A_Storage[:,41:,:] += (TP10series_q + ConAdj_y[Tenor_y==10][0]*use_convadj) #Intercept of extrapolation: add 10y premium with convexity adjustment...
    blocks = 8; size=int((201-41)/blocks)
    if use10y5yslope==True:
        Slope = ( TP10series_q + ConAdj_y[Tenor_y==10][0]*use_convadj  - (TP05series_q + ConAdj_y[Tenor_y==5][0]*use_convadj) )/(10-5)
    else: #Use instead the 2y-10y slope as initially mentioned in the paper.
        Slope = ( TP10series_q + ConAdj_y[Tenor_y==10][0]*use_convadj  - (TP02series_q + 0) )/(10-2)
    for block in range(blocks): #Do ten-year or 5-year horizon blocks to save on GPU memory space, at cost of minor speed deterioration. 
        start =  41+size*block; end = 41+size*(block+1)
        A_Storage[:,start:end,:] += (xp.asarray(Tenor_q[start:end])  - 10) @ Slope
    ConAdj_q = np.interp(Tenor_q, Tenor_y, ConAdj_y)
    if plot_conv == True:
        fig1 = plt.figure(figsize=[16/1.1, 9/1.1]); plt.plot( Tenor_q , 100*ConAdj_q, '.', Tenor_y, 100*ConAdj_y, '*'); plt.legend(('Interpolated Conv. Adjustment', 'Given Conv. Adjustment'), loc='best'); plt.xlabel('Conv. Adjustment at tenor (years)'); plt.ylabel('basis points'); plt.title('Interpolation of Convexity Adjustments'); #plt.show(block=False); plt.pause(0.001)
    if use_convadj==True:
        A_Storage[:,41:,:] -= xp.asarray(np.expand_dims(np.atleast_2d(ConAdj_q[41:]), 0))
    if replicateBeltonTP == True: #For exact replication, need to shift levels of TP, since we are not using the same convexity adjustments and same coefficients from ACM.
        Tenor_y =     np.array([     1,    2,    3,    5,    7,   10,   20,   30,   50])
        Adj_y =       np.array([ -0.05,-0.02,-0.10,-0.08,-0.08,-0.08,-0.25,-0.45,-0.15])
        Adj_q = xp.asarray(np.interp(Tenor_q, Tenor_y, Adj_y), dtype=xp.float32) 
        A_Storage += xp.reshape(Adj_q, (1,-1,1)) 
    #A_Storage[:,:,41:,:] -= xp.asarray(np.expand_dims(np.atleast_3d(ConAdj_q[41:]).T, 3))

def MakeOnTheRun(A_Storage): 
    Tenor_y =     np.array([    0.25,    2,    3,    5,    7,   10,   20,   30,   50])
    OntheRun_y =  np.array([   -0.07,-0.02,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04])
    Tenor_q = np.reshape(np.arange(0,50.25,0.25, dtype=np.float32), (-1,1))
    OntheRun_q = xp.asarray(np.interp(Tenor_q, Tenor_y, OntheRun_y), dtype=xp.float32)
    A_Storage += xp.reshape(OntheRun_q, (1,-1,1))  

def MakeZCBRates(A_NomsRates):
    A_NomsRates[:,1,...] = 1/(1+A_NomsRates[:,1,...]/400)  # Transform 1st par (=zcb) rate into 1st ZCB discount factor (do this for 0th par rate as well, inconsequential.)
    for T in range(2,A_NomsRates.shape[1]):                  # Get all other ZCB discount factors recursively "stripping the par curve":
        A_NomsRates[:,T,...] = (1 -  (A_NomsRates[:,T,...]/400)*xp.sum(A_NomsRates[:,1:T,...], axis=1)   ) /(1+ A_NomsRates[:,T,...]/400)
    A_NomsRates[:,1:,...] **= xp.reshape(-1/xp.arange(1,A_NomsRates.shape[1],dtype=xp.float32), tuple([1,-1] + [1 for x in  range(A_NomsRates.ndim-2)]) )   #Invert ZCB discount factors to get ZCB rates
    A_NomsRates[:,1:,...] -= 1 
    A_NomsRates[:,1:,...] *= 400   

def F_addLRP(Storage):
    n_exp_hor = Storage.shape[1]
    Tenor_y = np.array([2,5,10,30])
    LRP_y = np.array([45,32,24,19])/100
    Tenor_q = np.arange(0,n_exp_hor*0.25,0.25)
    LRP_q = np.interp(Tenor_q, Tenor_y, LRP_y) 
    Storage += xp.reshape(xp.asarray(LRP_q), (1,-1,1)) 

def F_addFRP(Storage):
    n_exp_hor = Storage.shape[1]
    Tenor_y = np.array([2,5])
    FRP_y = np.array([10,26])/100
    Tenor_q = np.arange(0,n_exp_hor*0.25,0.25)
    FRP_q = np.interp(Tenor_q, Tenor_y, FRP_y) 
    Storage += xp.reshape(xp.asarray(FRP_q), (1,-1,1)) 

# Plot Simulations
def PlotSims2(A_SimObs, A_SimSta, A_NomsRates, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP): 
    D_VarsToPlot = {'UGAP':(-2,2), 'Inflation':(0,4),'R Star':(0,2) ,'Fed Funds':(0,7), 'PRI':(-6,3), 'Int. Cost/GDP':(1,4), 'Tot. Balance/GDP':(-8,3), 'At Par Coupon Rates':(1,4),'TP2':(-1,0.5), 'TP10':(-1,1.5), 'G.state':(0,3), 'Z.state':(-1.5,0), 'Debt/GDP':(60,120),'Rollovers/GDP':(10,30),'WAC':(1,5),'WAM':(5,7) }
    L_VarsToPlot = list(D_VarsToPlot.keys()); axlims = list(D_VarsToPlot.values()) 
    n_VarstoPlot = len(L_VarsToPlot)
    n_TotCol = 4
    n_TotRow = np.ceil(n_VarstoPlot/n_TotCol).astype(int)
    M_CRPaths = xp.mean(A_NomsRates, axis=2)  #Only interested in plotting mean rates, so don't bring all array back from GPU
    if xp != np: # Bring arrays back from GPU, only for the intial state vector that we want to plot
        Arrays = [x.get() for x in [A_SimObs, A_SimSta, M_CRPaths, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP] ]
    else: 
        Arrays = [x       for x in [A_SimObs, A_SimSta, M_CRPaths, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP] ]
    A_SimObs, A_SimSta, M_CRPaths, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP = Arrays[:]; del Arrays
    L_StaNam = ['U.state', 'G.state','Z.state', 'P.state','F.state','Ulag.state','Glag.state','Zlag.state','Plag.state','Flag.state','EpsCPI.state','EpsPRI.state','EpsTP10.state','EpsTP2.state']  
    L_ObsNam = ['UGAP', 'Inflation', 'Fed Funds','CPI','TP10', 'TP2', 'PRI']
    fig2, axes = plt.subplots(nrows= n_TotRow, ncols= n_TotCol , sharex=True, sharey='none', figsize=(16/1.1,9/1.1))
    Custom=(cycler(color=['lightblue', 'lightskyblue', 'steelblue', 'lightskyblue','lightblue']) + cycler(lw=[1, 2, 3, 2, 1]))
    for Var in range(n_VarstoPlot):
        row = np.int32(Var/n_TotCol); col = Var % n_TotCol
        axes[row, col].set_prop_cycle(Custom)
        VarName = L_VarsToPlot[Var]
        axes[row, col].set_title(VarName)
        if (VarName in L_ObsNam):   A = A_SimObs[:,L_ObsNam.index(VarName),:]
        elif (VarName in L_StaNam): A = A_SimSta[:,L_StaNam.index(VarName),:]
        elif (VarName == 'R Star'): A = A_SimSta[:,L_StaNam.index('G.state'),:] + A_SimSta[:,L_StaNam.index('Z.state'),:]
        elif (VarName == 'Int. Cost/GDP'):    A = 400*A_IRCost/A_NGDP #Annualized: multiply by 4
        elif (VarName == 'Tot. Balance/GDP'): A =  A_SimObs[:,L_ObsNam.index('PRI'),:] - 400*A_IRCost/A_NGDP
        elif (VarName == 'Rollovers/GDP'):    A = 400*(A_DbtSvc - A_IRCost)/A_NGDP
        elif (VarName == 'Debt/GDP'): A = 100 * TotDebt/A_NGDP 
        elif (VarName == 'WAM'):      A = WAM
        elif (VarName == 'WAC'):      A = 400*A_IRCost/TotDebt
        if VarName!= 'At Par Coupon Rates': lines = axes[row, col].plot(np.percentile(A, np.array([15,30,50,70,85]), axis=1).T)
        else: qtr_tenors = [4*yr_tenor for yr_tenor in[1,2,3,4,5,7,10,20,30,50]];  axes[row, col].set_prop_cycle(cycler(color=[plt.cm.get_cmap('rainbow')(x/len(qtr_tenors)) for x in range(1,len(qtr_tenors)+1)])); lines = axes[row, col].plot(M_CRPaths[:,qtr_tenors])
        ylims = axes[row, col].set_ylim(axlims[Var])
    plt.legend(iter(lines), ('15%', '30%', 'median', '70%', '85%'), loc='upper right'); #plt.show(block=False)

def PlotRates(M_avgFFPaths, M_CRPaths): # Please provide already averaged across simualtions:  M_IRPaths = xp.mean(A_IRPaths, axis=2)
    M_IRPaths = xp.copy(M_CRPaths) # Create a copy (and not a view)...
    MakeZCBRates(M_IRPaths) # on which to apply in-place transformation to ZCB rates. 
    qtr_tenors = [4*yr_tenor for yr_tenor in[1,2,3,4,5,7,10,20,30,50]]
    if xp != np: 
        M_IRPaths, M_avgFFPaths, M_CRPaths= M_IRPaths.get(), M_avgFFPaths.get(), M_CRPaths.get()
    Arrays = [x[:,qtr_tenors] for x in [M_IRPaths, M_avgFFPaths, M_IRPaths-M_avgFFPaths, M_CRPaths, M_avgFFPaths,M_CRPaths-M_avgFFPaths ]]
    fig3, axes = plt.subplots(nrows= 4, ncols= 3 , sharex=True, sharey='none', figsize=(16/2,9/2)) 
    for r in [0,1]:
        for s in [0,1,2]:
            axes[r*2, s].set_prop_cycle(cycler(color=[plt.cm.get_cmap('rainbow')(x/len(qtr_tenors)) for x in range(1,len(qtr_tenors)+1)]))
            lines = axes[r*2, s].plot(Arrays[r*3+s]); 
            ylims = axes[r*2, s].set_ylim([(1,4),(1,4),(-1,2)][s])
            title = axes[r*2, s].set_title([['ZCB Rates','At Par Coupon Bonds rates'][r],'Avg. Exp. Future FFunds','Term Premium'][s])
        leg=axes[1,2].legend(iter(lines), [str(int(q/4))+' Yrs Maturity'for q in qtr_tenors  ], ncol=2, loc='upper right') 
        for s in [0,1]:
            title = axes[r*2+1, s].set_title(str(2+s*8) + 'y Rate Decomposition'); 
            lines = axes[r*2+1, s].plot( Arrays[0+r*3][:,1+s*5], 'black', Arrays[1+r*3][:,1+s*5], 'gold', Arrays[2+r*3][:,1+s*5], 'magenta' )
            ylims = axes[r*2+1, s].set_ylim((-1,4))
        leg=axes[3,2].legend(iter(lines),['Interest Rate','Avg. Fut. Exp. FFunds','Term Premium'], loc='upper right') ; axes[r*2+1,2].set_axis_off(); #plt.show(block=False)

# Wrapper for memory management, does entire rates block and overwrites intermediate results (Fed Fund rates, TPremia, ZCB rates) and only returns rates on Coupon Bonds isued at par. Averages of intermediate results can be shown setting plot_rates=True
def F_SimRat(A_SimSta, A_SimObs, ModelMats, n_exp_horizon=201, plot_rates=True, plot_coeff = True, plot_conv = True, use10y5yslope=True, use_convadj=True, plot_IRPcoeff=True, plot_IRP = True, estimate_not_assume=True, TP_is_for_ZeroCurve = True, replicateBeltonTP=True): 
    A_Storage = xp.empty((A_SimSta.shape[0], n_exp_horizon, A_SimSta.shape[2] ), dtype=xp.float32) 
    A_CPIPaths = xp.empty((A_SimSta.shape[0], 30*4+1, A_SimSta.shape[2] ), dtype=xp.float32) 
    MakeFFPaths3(A_SimSta, ModelMats, A_Storage, A_CPIPaths)
    M_avgFFPaths = xp.mean(A_Storage, axis=2)  # Small extra storage for plotting with plot_rates
    A_ExpFFR_05_10 = xp.copy(A_Storage[:,[4*5,4*10],:]) # Again, extra storage for expected FFR before overwriting them when adding TP to them.     
    A_FRNRates = xp.zeros((A_SimSta.shape[0], 1+5*4, A_SimSta.shape[2] ), dtype=xp.float32)  #Create storage for FRN rates
    A_FRNRates += xp.expand_dims(A_Storage[:,1,:],1);  F_addFRP(A_FRNRates) #Set FRN rate equal to 3 month expected FFR plus FRP premium.
    MakeTPPaths2(A_SimObs, A_Storage, plot_coeff = plot_coeff, plot_conv = plot_conv, use10y5yslope=use10y5yslope, use_convadj=use_convadj, estimate_not_assume=estimate_not_assume, TP_is_for_ZeroCurve = TP_is_for_ZeroCurve, replicateBeltonTP=replicateBeltonTP) #Notice it also transforms to par coupon rates
    if plot_rates==True : 
        PlotRates(M_avgFFPaths, xp.mean(A_Storage, axis=2))
    A_StoreIRP = F_MakeIRP(A_ExpFFR_05_10, A_SimSta, A_CPIPaths, plot_IRPcoeff=plot_IRPcoeff, plot_IRP = plot_IRP) #Get inflation risk premium for horizons up to 30y: Store = IRP
    A_StoreIRP *= -1                      # Flip sign:           Store  =    - IRP
    A_StoreIRP += A_Storage[:,:30*4+1,:]  # Add nominal yields:  Store =  (ExpNom + TP) - IRP
    A_StoreIRP -= A_CPIPaths[:,:30*4+1,:]; F_addLRP(A_StoreIRP); # Subtract expected inflation and add LRP to get tips yield: Store = ExpNom - ExpInf + TP - IRP + LRP  = (ExpNom-ExpINF) + (TP - IRP - FRP ) + FRP + LRP =  R + RRP + FRP + LRP = TIPS 
    return A_Storage, A_StoreIRP, A_FRNRates 

def MakeGDPPaths(Init_GDP, A_SimSta): # Notice that fixing initial price level as base, Init_GDP = Init_RGDP.    
    Init_Pot_RGDP = Init_GDP / (1 + 2*A_SimSta[0,0,0]/400) # POTRGDP = RGDP / (1 + RGDPGAP%), where RGDPGAP=2 x Unemployment Gap by Okun Law. Take 0th simulation UGAP wlog since in period zero all simulations start at assumed initial state. 
    A_Pot_RGDP = Init_Pot_RGDP * xp.cumprod(1+(A_SimSta[:,1,:])/400, axis=0)  # Potential RGDP quarterly growth is G state. 
    A_RGDP = A_Pot_RGDP * (1 + 2*A_SimSta[:,0,:]/400)                         # Apply Okun Law again to get RGDP paths
    A_NGDP = A_RGDP * xp.cumprod(1+(A_SimSta[:,3,:])/400, axis=0)             # Multiply by cumulative inflation to get nominal GDP
    return A_NGDP 

def F_MakeDebtStorages(n_period,n_exp_horizon,n_simula): # Prepares the storage spaces for Debt Block
    A_NomsFV     = xp.zeros((n_exp_horizon,n_simula), dtype=xp.float32)         # Tracks profiles of FVs for Nominals.  Shape n_exp_horizon x n_simula                                              
    A_TipsFV     = xp.copy(A_NomsFV)                                            # Tracks profiles of FVs for TIPS.      Shape n_exp_horizon x n_simula
    A_TipsFVadj  = xp.copy(A_NomsFV)                                            # Tracks profiles of inflation-adjusted FVs for TIPS. Shape n_exp_horizon x n_simula
    A_TipsFVmax  = xp.copy(A_NomsFV)                                            
    A_TipsFVmaxOLD = xp.copy(A_NomsFV) 
    A_FrnsFV     = xp.copy(A_NomsFV)                                            # Tracks profiles of FVs for FRNS.      Shape n_exp_horizon x n_simula                                         
    A_IRCost     = xp.zeros((n_period,n_simula), dtype=xp.float32)              # Tracks interest cost from coupons generating cash flows. Shape n_period x n_simula
    A_TipsFVCost = xp.copy(A_IRCost)                                            # Tracks interest cost that TIPS accrue (without cash flow) at time t as inflation/deflation of their principal. Shape n_period x n_simula
    A_DbtSvc     = xp.copy(A_IRCost)                                            # Tracks debt service cost, sum of coupon cost and maturing face values (for tips, maturing is max(FV, adjFV)).  Shape n_period x n_simula
    A_TotDfc     = xp.copy(A_IRCost)                                            # Tracks financing need, sum of debt service cost (above) plus primary deficit. Shape n_period x n_simula
    Avg_IssRate  = xp.zeros(n_period, dtype=xp.float32)                         # Tracks average issuance coupon rate. Shape n_period
    A_TotCoup    = xp.zeros((n_exp_horizon,2,n_simula), dtype=xp.float32)       # Tracks total coupons at every tenor from Nominals and Inflation-adjsuted coupons from tips.  Shape n_exp_horizon x 2 x n_simula
    Store_Pvals  = xp.zeros((n_exp_horizon-1,n_simula), dtype=xp.float32)       # Computes sum of TIPS and Noms par FVs (=PVs) to get TOTdebt and WAM
    TotDebt      = xp.copy(A_IRCost)
    WAM          = xp.copy(A_IRCost)
    del n_period,n_exp_horizon,n_simula
    return locals().copy()

def Performance(Init_DebtProfiles, RateStorages, A_SimObs, A_NGDP, Securities, Const_and_MEVs=None, M_Kernels=None, CoeffstoConst_and_MEVs=None, SingleIssuance = False, Static = False, Dynamic = False, QuartersperCoup=1):
    assert (SingleIssuance  + Static  + Dynamic == 1), "Set to True one and only one option: SingleIssuance, Stati, Dynamic"  
    n_securi = Securities.shape[1]
    if SingleIssuance == True: N_strats = n_securi
    else: N_strats = CoeffstoConst_and_MEVs.shape[2]
    Avg_IssRate = xp.zeros(N_strats, dtype= xp.float32)
    Avg_IRCost = xp.zeros(N_strats, dtype= xp.float32)
    Std_IRCost = xp.zeros(N_strats, dtype= xp.float32)
    Std_TotBal = xp.zeros(N_strats, dtype= xp.float32)
    Cor_IRC_PRI = xp.zeros(N_strats, dtype= xp.float32)
    DebtStorages = F_MakeDebtStorages(n_period,n_exp_horizon,n_simula)
    IssuanceStrat = xp.zeros((n_period, n_securi, n_simula), dtype = xp.float32)
    UnadjustedKernelIssuance, SumUnadjustedKernelIssuance = None, None
    for i in range(N_strats):
        if SingleIssuance == True:
            IssuanceStrat*=0; IssuanceStrat[:,i,:]=1
        elif Static == True:
            IssuanceStrat[:,:,:] = xp.tile( xp.expand_dims(M_Kernels @ CoeffstoConst_and_MEVs[:,:,i], axis=0),  (n_period, 1, n_simula)   )
        elif Dynamic == True: #Dynamic: Kernels coefficients give dollars of issuance, not shares.  IssuanceStrat will be overwritten
            UnadjustedKernelIssuance =   M_Kernels[:,1:]    @  (CoeffstoConst_and_MEVs[:,:,i] @ Const_and_MEVs) # Dollar Issuance caused by Kernelse else than Baseline   
            SumUnadjustedKernelIssuance = xp.sum(UnadjustedKernelIssuance, axis = 1) 
            #F_Kernels2Strategy(IssuanceStrat, Const_and_MEVs, M_Kernels, CoeffstoConst_and_MEVs[:,:,i], static = static)
        #MakeDbtPaths1(Init_DbtFVout, Init_AvgCoupRate, Init_TipsFVadj, Init_FrnsFV, IssuanceStrat, NomsPos,  TipsPos,  FrnsPos, NomsTenors, TipsTenors, FrnsTenors, MaxFrnsTen, A_NomsRates_view, A_TipsRates_view, A_FrnsRates_view, A_SimObs, A_NGDP, *list(DebtStorages.values()), M_Kernels, UnadjustedKernelIssuance, SumUnadjustedKernelIssuance,TrackWAM=False, Dynamic = Dynamic, QuartersperCoup=QuartersperCoup )
        MakeDbtPaths1(*Init_DebtProfiles, IssuanceStrat, *RateStorages, A_SimObs, A_NGDP, *list(DebtStorages.values()), M_Kernels, UnadjustedKernelIssuance, SumUnadjustedKernelIssuance,TrackWAM=False, Dynamic = Dynamic, QuartersperCoup=QuartersperCoup )
        DebtStorages['A_IRCost'] /= A_NGDP
        DebtStorages['A_IRCost'] *= 400
        Axis = 1 #Select 1 to compute statistics across simulations for a fixed period, then average across periods. Select 0 to do the converse: compute stats across periods for a fixed simulation, then average across simulations. 
        Startperiod =79 #Starting period for statistics window. Select a number from 1 to 79 (or -1)    
        Avg_IssRate[i] = xp.mean(DebtStorages['Avg_IssRate'][Startperiod:])
        Avg_IRCost[i] = xp.mean(xp.mean(DebtStorages['A_IRCost'][Startperiod:,:], axis= Axis ))
        Std_IRCost[i] = xp.mean(xp.std( DebtStorages['A_IRCost'][Startperiod:,:], axis= Axis )) #xp.mean(xp.std( A_IRCost[1:,:], axis=0 ))
        Std_TotBal[i] = xp.mean(xp.std(-A_SimObs[Startperiod:,6,:] + DebtStorages['A_IRCost'][Startperiod:,:], axis=Axis )) #xp.mean(xp.std(-A_SimObs[1:,6,:] + A_IRCost[1:,:], axis=0 ))
        Cor_IRC_PRI[i] = xp.mean(  xp.mean((DebtStorages['A_IRCost'][Startperiod:,:] - xp.mean(DebtStorages['A_IRCost'][Startperiod:,:], axis=Axis, keepdims=True))*(-A_SimObs[Startperiod:,6,:] - xp.mean(-A_SimObs[Startperiod:,6,:], axis=Axis, keepdims=True)), axis=Axis)/(xp.std(DebtStorages['A_IRCost'][Startperiod:,:], axis=Axis)*xp.std(-A_SimObs[Startperiod:,6,:], axis=Axis))  )
    return Avg_IssRate, Avg_IRCost, Std_IRCost, Std_TotBal, Cor_IRC_PRI

def MakeDbtPaths1(Init_DbtFVout, Init_AvgCoupRate, Init_TipsFVadj, Init_FrnsFV, IssuanceStrat, NomsPos,  TipsPos,  FrnsPos, NomsTenors, TipsTenors, FrnsTenors, MaxFrnsTen, A_NomsRates_view, A_TipsRates_view, A_FrnsRates_view, A_SimObs, A_NGDP, A_NomsFV, A_TipsFV, A_TipsFVadj, A_TipsFVmax, A_TipsFVmaxOLD, A_FrnsFV, A_IRCost, A_TipsFVCost, A_DbtSvc, A_TotDfc, Avg_IssRate, A_TotCoup, Store_Pvals, TotDebt, WAM, M_Kernels=None, UnadjustedKernelIssuance=None, SumUnadjustedKernelIssuance=None,TrackWAM=False, Dynamic = False, QuartersperCoup=1): 
    n_period = A_NomsRates.shape[0]                                                 # For clarity
    A_NomsFV[:,:]    = xp.reshape(Init_DbtFVout[:,0],(-1,1))                      # Initial profile of Nominal debt outstanding face values
    A_TipsFV[:,:]    = xp.reshape(Init_DbtFVout[:,1],(-1,1))                      # Initial profile of TIPS debt outstanding face values, NOT INFLATION ADJUSTED
    A_TipsFVadj[:,:] = xp.reshape(Init_TipsFVadj    ,(-1,1))                      # Initial profile of TIPS debt outstanding face values, CPI INFLATION ADJUSTED
    A_FrnsFV[:,:]    = xp.reshape(Init_FrnsFV       ,(-1,1))                      # Initial profile of FRNs debt outstanding face values    
    A_TotCoup[:,0,:] = xp.reshape(Init_DbtFVout[:,0]*xp.nan_to_num(Init_AvgCoupRate[:,0])/400,(-1,1)) # Initial value of coupons on Nominal securities
    A_TotCoup[:,1,:] = xp.reshape(Init_TipsFVadj    *xp.nan_to_num(Init_AvgCoupRate[:,1])/400,(-1,1)) # Initial value of INFLATION ADJUSTED coupons on TIPS securities
    PriDeficit = - (A_SimObs[:,6,:]/400)*A_NGDP
    CPIInfl = xp.expand_dims(1+A_SimObs[:,3,:]/400, 0) 
    Index = xp.copy(A_IRCost); Index[0,:] = 1; BaseDebt = xp.sum(A_NomsFV[1:,:] + A_TipsFVmax[1:,:] + A_FrnsFV[1:,:], axis=0)
    for t in range(0,n_period):
        A_TotCoup   = xp.roll(A_TotCoup  , -1, axis=0); A_TotCoup[-1,:,:] = 0   
        A_NomsFV    = xp.roll(A_NomsFV   , -1, axis=0); A_NomsFV[-1,:]    = 0        # Update current profile of debt FV before new issuance to be old debt shifted one tenor.  
        A_TipsFV    = xp.roll(A_TipsFV   , -1, axis=0); A_TipsFV[-1,:]    = 0     
        A_TipsFVadj = xp.roll(A_TipsFVadj, -1, axis=0); A_TipsFVadj[-1,:] = 0 
        A_FrnsFV    = xp.roll(A_FrnsFV   , -1, axis=0); A_FrnsFV[-1,:]    = 0   
        A_TotCoup[:,1,:]*= CPIInfl[:,t,:]                                        # Coupons of TIPS are inflation adjusted. Update with current inflation
        A_TipsFVmaxOLD   = xp.maximum(A_TipsFV,A_TipsFVadj)                         # Keep a copy before updating TIPS adjusted face value by new inflation adjustment
        A_TipsFVadj     *= CPIInfl[:,t,:]                                           # Update adjusted FV by inflation
        A_TipsFVmax      = xp.maximum(A_TipsFV,A_TipsFVadj)                            # This is the FV effectively coming due.
        xp.sum(A_TipsFVmax  - A_TipsFVmaxOLD, axis=0, out=A_TipsFVCost[t, :])         # Keep track of inflation of maximum between face value and adjusted face value as an interest cost. 
        A_IRCost[t,:] = QuartersperCoup * xp.sum(A_TotCoup[:-1:QuartersperCoup,:,:], axis=(0,1)) + xp.sum(A_FrnsFV[:MaxFrnsTen+1,:]*A_FrnsRates_view[t,:,:]/400, axis=0) # Coupon Cost is sums coupons generated by all future (even or odd) tenors of last period, both coupons from nominal securities and adjusted coupons from TIPS.  
        A_DbtSvc[t,:] = A_IRCost[t,:] + A_NomsFV[0,:] + A_TipsFVmax[0,:] + A_FrnsFV[0,:]       # Debt service is coupon expense plus FV rolled, i.e. FV that was scheduled last periods to be due next period
        A_TotDfc[t,:] = PriDeficit[t,:] + A_DbtSvc[t,:]                                        # Total deficit to be financed with PV debt is Debt Sevice + Primary Deficit = Coupon Interest Expense + Maturing Face value + Primary Deficit
        if Dynamic == False:  
            NewIssuance =       IssuanceStrat[t,:,:] * xp.expand_dims(A_TotDfc[t,:], axis=0)       # Deficit to be financed with issuance is split across tenors according to issuance strategy   
        else: 
            AddIRCost =   xp.expand_dims(M_Kernels[:,1:] @ CoeffstoConst_and_MEVs[:,3], axis=1 ) @ xp.expand_dims(A_IRCost[t,:], axis=0) #Add QUARTERLY, non-annualized IRCost to Deficit MEV. 
            NewIssuance = xp.expand_dims(M_Kernels[:,0], axis=1)* xp.expand_dims(A_TotDfc[t,:] - (SumUnadjustedKernelIssuance[t,:]+xp.sum(AddIRCost, axis=0))*Index[t,:], axis=0) + (UnadjustedKernelIssuance[t,:,:]+ AddIRCost )*xp.expand_dims(Index[t,:], axis=0); 
            IssuanceStrat[t,:,:] = NewIssuance/xp.expand_dims(A_TotDfc[t,:], axis=0) # Keep a record of issuance shares. 
        A_NomsFV[NomsTenors,:]    +=  NewIssuance[NomsPos,:]                                   # plus 2) new Issuance of FV (from +1qtr forward), equal to share of to-be-financed PV for each tenor (FV=PV since issued at par) 
        A_TipsFV[TipsTenors,:]    +=  NewIssuance[TipsPos,:]  
        A_TipsFVadj[TipsTenors,:] +=  NewIssuance[TipsPos,:]  
        A_FrnsFV[FrnsTenors,:]    +=  NewIssuance[FrnsPos,:]  
        A_TotCoup[NomsTenors,0,:] +=  (NewIssuance[NomsPos,:]*A_NomsRates_view[t,:,:]/400)        
        A_TotCoup[TipsTenors,1,:] +=  (NewIssuance[TipsPos,:]*A_TipsRates_view[t,:,:]/400)        # Add new TIPS issuance x TIPS coupon rate to TIPS INFATION ADJUSTED coupons.
        Avg_IssRate[t] = xp.mean(xp.sum(A_NomsRates_view[t,:,:] * IssuanceStrat[t,NomsPos,:], axis=0) + xp.sum(A_TipsRates_view[t,:,:] * IssuanceStrat[t,TipsPos,:], axis=0)  + xp.sum(A_FrnsRates_view[t,FrnsTenors,:] * IssuanceStrat[t,FrnsPos,:], axis=0) )
        Store_Pvals  =  A_NomsFV[1:,:] + A_TipsFVmax[1:,:] + A_FrnsFV[1:,:]
        TotDebt[t,:] = xp.sum(Store_Pvals, axis=0)
        if ((Dynamic == True) and (t < n_period-1)): Index[t+1,:] = xp.sqrt(TotDebt[t,:] / BaseDebt)   #Debt stock growth index to inflate the issuence amount with Kernels other than Baseline in Dynamic strategies. 
        if TrackWAM == True:
            Store_Pvals  /= xp.expand_dims(TotDebt[t,:], axis=0) #Get distribution of share of PValue of debt across tenors. 
            WAM[t,:]     =  0.25 * np.squeeze(xp.reshape(xp.arange(1,n_exp_horizon,dtype=xp.int32),(1,-1)) @ Store_Pvals)
    A_IRCost += A_TipsFVCost #Add to the cash flow cost the accrued one

#Plotter of Strategies' risk-cost tradeoff. 
def PlotStrats(Avg_IssRate, Avg_IRCost,Std_IRCost,Std_TotBal, Cor_IRC_PRI, Title, StratNames=[], ColorList=None, M_style='o', Y_lim=(None,None) , X_lim=(None,None), XX_lim=(None,None), Tabulate=True ):
    if xp!= np: X=Std_IRCost.get(); Y=Avg_IRCost.get(); XX=Std_TotBal.get(); XXX=Cor_IRC_PRI.get(); Z=Avg_IssRate.get()
    else : X=Std_IRCost; Y=Avg_IRCost; XX=Std_TotBal; XXX=Cor_IRC_PRI; Z=Avg_IssRate
    fig4, axes = plt.subplots(nrows= 1, ncols= 2 , sharex='none', sharey=True, figsize=(16/4,9/3))
    axes[0].scatter(X,Y, c=ColorList, marker=M_style); axes[0].set_xlabel("Std IR Cost"); axes[0].set_ylabel("Avg IR Cost"); axes[0].set_title(Title + " AvG int cost vs StD int cost",fontsize=9);axes[0].set_ylim(Y_lim); axes[0].set_xlim(X_lim)
    axes[1].scatter(XX,Y,c=ColorList, marker=M_style);axes[1].set_xlabel("Std Deficit"); axes[1].set_ylabel("Avg IR Cost"); axes[1].set_title(Title + " AvG int cost vs StD (int cost + pri. deficit)",fontsize=9); axes[1].set_xlim(XX_lim) 
    for i, label in enumerate(StratNames): 
        axes[0].annotate(label, (X[i], Y[i]))
        axes[1].annotate(label, (XX[i], Y[i]))
    if Tabulate==True:
        row_names = ["Average Issuance Rate", "Average IR Cost/GDP  ","StDev IR Cost/GDP  ","StDev (IR Cost/GDP + PRI Deficit/GDP)  ", "Corr (IRCost/GDP , PRI Deficit/GDP)"]
        import pandas as pd
        print(pd.DataFrame([np.round(Z,2), np.round(Y,2),np.round(X,2),np.round(XX,2), np.round(XXX,2)],  row_names, [name[0:5]  for name in StratNames]))

def F_MakeRateStorages(Securities, A_NomsRates, A_TipsRates, A_FrnsRates):
    NomsWhere,  TipsWhere,  FrnsWhere =  [  Securities[0,:]==x  for x in [0,1,2]][:]
    NomsTenors, TipsTenors, FrnsTenors = [4*Securities[1,x].astype(np.int32) for x in [NomsWhere,TipsWhere,FrnsWhere]][:]
    NomsPos,       TipsPos,    FrnsPos = [np.arange(len(x))[x] for x in [NomsWhere,  TipsWhere,  FrnsWhere]][:]
    MaxFrnsTen = np.max(np.append(FrnsTenors,-1))
    A_NomsRates_view, A_TipsRates_view, A_FrnsRates_view = A_NomsRates[:,NomsTenors,:], A_TipsRates[:,TipsTenors,:], A_FrnsRates[:,:MaxFrnsTen+1,:] #Prepare views in advance.
    return NomsPos,  TipsPos,  FrnsPos, NomsTenors, TipsTenors, FrnsTenors, MaxFrnsTen, A_NomsRates_view, A_TipsRates_view, A_FrnsRates_view


# Use functions. 

# Replicate Figure 4: Macro, Fiscal, Rates and Debt blocks
ModelMats = F_BeltonMatrices(V_ParSrt)
if ReplicateBelton ==  True: V_StaSrt, Init_GDP = np.array([-0.5,1.5,-1,1.8, 2   ,-0.5,1.5,-1,1.8 ,2   ,0,0,-0.25,0]), 19882 # Initial states, implied by Belton et al Fig 4, and zero initial AR1 errors except for EpsTP10 at -0.25. # Q4 2017 Nominal GDP, in current Billon USD
else:                        V_StaSrt, Init_GDP = F_InitStates(startyear,startquarter)    
A_SimSta = F_SimSta(n_period, n_simula, V_StaSrt, ModelMats) 
A_SimObs = F_SimObs(A_SimSta, ModelMats)
A_NomsRates, A_TipsRates, A_FrnsRates = F_SimRat(A_SimSta, A_SimObs, ModelMats, plot_rates=plotFigs, plot_coeff = plotFigs, plot_conv = plotFigs, use10y5yslope=use10y5yslope, use_convadj=use_convadj, plot_IRPcoeff=plotFigs, plot_IRP = plotFigs, estimate_not_assume=estimate_not_assume, TP_is_for_ZeroCurve = True, replicateBeltonTP=replicateBeltonTP)
A_NGDP = MakeGDPPaths(Init_GDP, A_SimSta)
Init_DebtProfiles = F_InitiProfiles(startyear,startquarter, path_CRSP, plotFigs=plotFigs)
IssuanceStrat = xp.tile(xp.expand_dims(Kernel1_Baseline,0), (n_period,1,n_simula))  # Use Baseline Strategy
RateStorages = F_MakeRateStorages(Securities, A_NomsRates, A_TipsRates, A_FrnsRates)# Pre-slices Rates arrays, creates some indexes. 
DebtStorages = F_MakeDebtStorages(n_period,n_exp_horizon,n_simula)                  # Creates storage arrays  
MakeDbtPaths1(*Init_DebtProfiles, IssuanceStrat, *RateStorages, A_SimObs, A_NGDP, *list(DebtStorages.values()), TrackWAM=True, Dynamic = False, QuartersperCoup=QuartersperCoup)
PlotSims2(A_SimObs, A_SimSta, A_NomsRates, DebtStorages['A_IRCost'], DebtStorages['A_DbtSvc'], DebtStorages['A_TotDfc'], DebtStorages['TotDebt'], DebtStorages['WAM'], A_NGDP)

# Replicate Figure 5: Performance of single issuance strategies (and baseline) 
Avg_IssRate, Avg_IRCost, Std_IRCost, Std_TotBal, Cor_IRC_PRI =  Performance(Init_DebtProfiles, RateStorages, A_SimObs, A_NGDP, Securities, SingleIssuance = True, QuartersperCoup=QuartersperCoup)
StratNames = [ ['  ','T ', 'F '][int(s)] + str(t) + 'y' for s,t in zip(Securities[0,:], Securities[1,:] ) ]
ColorList =  [ ['blue','red', 'gray'][int(s)]  for s in  Securities[0,:] ]
TipsandFRN = max(Securities[0,:]) > 0
PlotStrats(Avg_IssRate, Avg_IRCost,Std_IRCost,Std_TotBal, Cor_IRC_PRI, 'Single Issuance Strategies', StratNames=StratNames, Y_lim=(2,4.5) , X_lim=(0+0.5*TipsandFRN,2+0.5*TipsandFRN), XX_lim=(1+0.8*TipsandFRN,3-0.2*TipsandFRN), ColorList=ColorList  )

#Replicate Figure 6: Performance of Kernel Strategies. 
N_StratVary = 20 # This controls how fine the variation into strategies is. 
M_Kernels = xp.concatenate((Kernel1_Baseline , Kernel2_Bills, Kernel3_Belly, Kernel4_Bonds), axis=1)
Const = xp.ones((n_period,1, n_simula),dtype=xp.float32)
CoeffstoConst = xp.zeros((M_Kernels.shape[1],1,N_StratVary*3+1), dtype=xp.float32); # Loading on each Kernel (or rows) for the value of Const and MeVs (on columns)
weight_K1 = 1
for s in range(N_StratVary*3+1):
    weight_K2 = (s>= 0*N_StratVary)*(s<1*N_StratVary)*(-0.4  +((s-0*N_StratVary)/(N_StratVary-1))*(0.5  - (-0.4 )))
    weight_K3 = (s>= 1*N_StratVary)*(s<2*N_StratVary)*(-0.08 +((s-1*N_StratVary)/(N_StratVary-1))*(0.1  - (-0.08)))
    weight_K4 = (s>= 2*N_StratVary)*(s<3*N_StratVary)*(-0.0  +((s-2*N_StratVary)/(N_StratVary-1))*(0.25 - (-0.0 )))
    CoeffstoConst[:,0,s] = xp.array([weight_K1, weight_K2,weight_K3 , weight_K4], dtype=xp.float32)             # Set Coefficients to Constant Only.
Avg_IssRate, Avg_IRCost, Std_IRCost, Std_TotBal, Cor_IRC_PRI =  Performance(Init_DebtProfiles, RateStorages, A_SimObs, A_NGDP, Securities, Const_and_MEVs=Const,  M_Kernels=M_Kernels, CoeffstoConst_and_MEVs=CoeffstoConst, Static=True, QuartersperCoup=QuartersperCoup)
ColorList = ['olivedrab' for x in range(N_StratVary)] + ['darkviolet' for x in range(N_StratVary)] + ['aquamarine' for x in range(N_StratVary)] + ['black']
StratNames = ['' for x in range(3*N_StratVary)] + ['Baseline']
StratNames[N_StratVary-1] = 'More Bills'; StratNames[2*N_StratVary-1] = 'More Belly'; StratNames[3*N_StratVary-1] = 'More Bonds'
PlotStrats(Avg_IssRate, Avg_IRCost,Std_IRCost,Std_TotBal, Cor_IRC_PRI, 'Vary Kernels', ColorList=ColorList, StratNames=StratNames, Y_lim=(2.5,3.8) , X_lim=(0.5,2), XX_lim=(1.5,3), Tabulate=False)

# Replicate Fig 13 and Fig 11 with Dynamic Strategy with Coefficients as in Fig 10
if ReplicateBelton ==  True:  # Need to re-compute macro block and rates block, as Fig 11,13 were done with different starting states. 
    del A_SimSta, A_SimObs, A_NomsRates, A_TipsRates, A_FrnsRates, RateStorages, DebtStorages #Free up as much space as possible
    V_StaSrt = np.array([-0.5,1.5,-1.5,1.75,1.15,-0.5,1.5,-1.5,1.75,1.15,0,0,-0.55,0])
    A_SimSta = F_SimSta(n_period, n_simula, V_StaSrt, ModelMats) 
    A_SimObs = F_SimObs(A_SimSta, ModelMats)
    A_NomsRates, A_TipsRates, A_FrnsRates = F_SimRat(A_SimSta, A_SimObs, ModelMats, plot_rates=False, plot_coeff = False, plot_conv = False, use10y5yslope=use10y5yslope, use_convadj=use_convadj, plot_IRPcoeff=False, plot_IRP = False, estimate_not_assume=estimate_not_assume, TP_is_for_ZeroCurve = True, replicateBeltonTP=True)
    RateStorages = F_MakeRateStorages(Securities, A_NomsRates, A_TipsRates, A_FrnsRates)
    DebtStorages = F_MakeDebtStorages(n_period,n_exp_horizon,n_simula)                  # Creates storage arrays 
Const_and_MEVs  = xp.concatenate((xp.ones((n_period,1, n_simula),dtype=xp.float32), (A_NomsRates[:,[2*4],:] - A_SimSta[:,[3],:]), A_SimObs[:,[4],:], - xp.expand_dims(A_SimObs[:,6,:] * A_NGDP /400, axis=1) ),    axis=1 ) # MEVS are: Constant, 2y Real Rates, 10y Term Premium, Deficit (Only PRI here, IRCost wll be added in Debt Loop). 
CoeffstoConst_and_MEVs =  xp.array([[-279.4 , 255.2, 418.1, -0.6],  # (Non normalized yet) betas of K2 to Constant and MEVs
                                    [   70.6,  -6.2,  70.3,  0.3],  # (Non normalized yet) betas of K3 to Constant and MEVs
                                    [  -42.3,  25.0,  -5.9, -0.1]]) # (Non normalized yet) betas of K4 to Constant and MEVs
#Figure 11:
UnadjustedKernelIssuance =   M_Kernels[:,1:]    @  (CoeffstoConst_and_MEVs @ Const_and_MEVs)
SumUnadjustedKernelIssuance = xp.sum(UnadjustedKernelIssuance, axis = 1) # Very small quantities, between -0.0000009 and + 0.0000008.
n_securi = Securities.shape[1]
IssuanceStrat = xp.zeros((n_period, n_securi, n_simula), dtype = xp.float32)
MakeDbtPaths1(*Init_DebtProfiles, IssuanceStrat, *RateStorages, A_SimObs, A_NGDP, *list(DebtStorages.values()), M_Kernels, UnadjustedKernelIssuance, SumUnadjustedKernelIssuance,TrackWAM=True, Dynamic = True, QuartersperCoup=QuartersperCoup)
if xp != np: X = IssuanceStrat[0,:,0].get(); XX = xp.mean(IssuanceStrat[79,:,:], axis=1).get()
else: X = IssuanceStrat[0,:,0]; XX = xp.mean(IssuanceStrat[79,:,:], axis=1)
row_names = ["Dynamic Issuance Strategy at Initial MEVs", "Mean Dynamic Issuance Strategy at long-run MEVs"]
SecNames  = [ ['  ','T ', 'F '][int(s)] + str(t) + 'y' for s,t in zip(Securities[0,:], Securities[1,:] ) ]
print(' ')
print(pd.DataFrame([np.round(X,2), np.round(XX,2)],  row_names, [name[0:4]  for name in SecNames]))

#Figure 13:
Avg_IssRateD, Avg_IRCostD, Std_IRCostD, Std_TotBalD, Cor_IRC_PRID =  Performance(Init_DebtProfiles, RateStorages, A_SimObs, A_NGDP, Securities, Const_and_MEVs=Const_and_MEVs,  M_Kernels=M_Kernels, CoeffstoConst_and_MEVs=xp.expand_dims(CoeffstoConst_and_MEVs, axis=2), Dynamic=True, QuartersperCoup=QuartersperCoup)
ColorList = ['olivedrab' for x in range(N_StratVary)] + ['darkviolet' for x in range(N_StratVary)] + ['aquamarine' for x in range(N_StratVary)] + ['black'] +['orange']
StratNames = ['' for x in range(3*N_StratVary)] + ['Baseline'] + ['Dynamic Optimal']
StratNames[N_StratVary-1] = 'More Bills'; StratNames[2*N_StratVary-1] = 'More Belly'; StratNames[3*N_StratVary-1] = 'More Bonds'
PlotStrats(xp.append(Avg_IssRate,Avg_IssRateD) , xp.append(Avg_IRCost, Avg_IRCostD) , xp.append(Std_IRCost, Std_IRCostD) , xp.append(Std_TotBal, Std_TotBalD), xp.append(Cor_IRC_PRI,Cor_IRC_PRID), 'Vary Kernels', ColorList=ColorList, StratNames=StratNames, Y_lim=(2,4) , X_lim=(0,2), XX_lim=(1,3), Tabulate=False)



toc=time()
print('Elapsed time: ' +str(round(toc-tic,2)) + 'secs')
plt.show()



