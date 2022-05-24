import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm.notebook import trange          # To show progress bars in Jupyter.
from cycler import cycler                       # To set customized plot colors.

try:
    import cupy as cp
    print('Cupy is installed, GPU will be used')
    xp = cp                 # Use Cupy if available in Cupy/Numpy agnostic functions.
    pool = cp.get_default_memory_pool(); pool.free_all_blocks() # Not using Cupy unified/managed memory pool: seems to have negative effect on GPU memory and speed.
    TotDedicMem=cp.cuda.Device(device=cp.cuda.runtime.getDevice()).mem_info[0] #Total free dedicated memory found on default device (May be below total memory...)
except ImportError:
    print('Cupy is not installed, GPU will not be used')
    xp = np                 # If Cupy is not available, fall back to Numpy

# Macro and Fiscal blocks
def F_Settings_to_Belton(D_Setup):
    if D_Setup["ReplicateBelton"]== True:
        D_Setup["start_year"] = 2017
        D_Setup["start_quarter"] = 4
        D_Setup["Securities"]= np.array([[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],  # 1st row specifying security type (Nom=0, TIPS=1, FRN=2),
                                         [1 , 2 , 3 , 5 , 7 , 10, 20, 30, 50]]) # 2nd row specifying tenor, in years (from 0.25 to n_exp_horizon//4, always multiples of 0.25)
        D_Setup["Kernel1_Baseline"] = xp.reshape(xp.array([0.475,  0.11,  0.09, 0.115, 0.08 , 0.085,   0.0, 0.045,    0.0], dtype= xp.float32), (-1,1))
        D_Setup["Kernel2_Bills"]    = xp.reshape(xp.array([ 1.00, -0.21, -0.17, -0.22, -0.16, -0.15,   0.0, -0.09,    0.0], dtype= xp.float32), (-1,1)) #Into Bills
        D_Setup["Kernel3_Belly"]    = xp.reshape(xp.array([-0.25,  0.25,  1.00,  0.50, -0.50, -0.75,   0.0, -0.25,    0.0], dtype= xp.float32), (-1,1)) #Into Belly
        D_Setup["Kernel4_Bonds"]    = xp.reshape(xp.array([ 0.00, -0.41, -0.33, -0.41, -0.10,  0.25,   0.0,  1.00,    0.0], dtype= xp.float32), (-1,1)) #Into Bonds
        D_Setup["CBO_weight"] = 1.0
        D_Setup["CBO_projection"] = xp.array([ -2.45], dtype= xp.float32) # Belton et al primary deficit in initial periods is constantat at 2.45%.
        D_Setup["No_TIPS_FRN"] = False
        D_Setup["n_period"] = 80
        D_Setup["n_simula"] = 2000
        D_Setup["n_exp_horizon"] = 201
        D_Setup["use10y5yslope"] = False
        D_Setup["use_convadj"] = True
        D_Setup["replicateBeltonTP"] = True
        D_Setup["ELB"] = 0.125
        D_Setup["QuartersperCoup"] = 1
        D_Setup["estimate_not_assume"] = True
        D_Setup["Supply_Effects"] = True
        D_Setup["L_ParNam"] =         ['rhoU1','betaUR','rhoU2','rhoZ1','betaPU','rhoP1','rhoP2','betaFTay','rhoF1','rhoEpsCPI','rhoEpsPRI','rhoEpsTP10','rhoEpsTP2','alphaZ','betaPPE2pct', 'sigmaU',  'sigmaG',  'sigmaZ' , 'sigmaP',  'sigmaNuCPI',  'sigmaNuPRI',  'sigmaNuTP10',  'sigmaNuTP2','ATP10', 'ATP2','alphaPRI', 'betaTP10U', 'BTP2U','betaTP10TP2','betaPRIU']
        D_Setup["V_ParSrt"] = np.array([  1.57,  0.028 ,  -0.62,  0.917,  -0.133,   0.58,   0.26,      0.15,   0.85,      0.295,       0.92,        0.73,      0.63,(1-0.917)*(-0.5),0.16*2,     0.24,    0.0624,      0.018,     0.79,          1.70,          0.35,           0.41,          0.09,   0.51,  -0.05,      0.34,       0.207,-0.014+0.42*0.207,0.42,     -1.5 ])
        D_Setup["V_ParSrt"][D_Setup["L_ParNam"].index('alphaPRI')] -=  0.4  # Belton et al long term deficit does not settle to the constant alpha_PRI = 0.34 but to -0.2 (my non-zero long term UGAP induced by ELB is not enough to account for all of this difference, only pushes down from 0.34 to 0.2)
        print("Settings reset to values for Belton et al replication")
    else:
        print("Settings left to user-defined values")

def F_InitStates(D_Setup):
    """
    F_InitStates initializes the model states and GDP for an arbitrary initial year and quarter. The model states are:
    UGAP: unemployment gap
    G: potential real growth, random walk component of the neutral rate R*
    Z: transitory component of R*, AR1.
    PI: PCE core inflation
    FFR: Fed Funds Rate
    + lag of the 5 states above
    EpsPRI:  AR1 residual for primary surplus
    EpsCPI:  AR1 spread between CPI and PCE
    EpsTP10: AR1 residual for 10y Term Premium
    EpsTP2:  AR1 residual for  2y Term Premium
    Also returs Init_GDP, the initial level of nominal GDP will be used to create GDP paths, in turned used for normalizations.
    """
    if D_Setup["ReplicateBelton"] ==  True: # Only initial states consistent with Fig 4.Close but note exactly equal to my retrieved states for 2017Q4.
        V_StaSrt, Init_GDP = np.array([-0.5,1.5,-1,1.8, 2   ,-0.5,1.5,-1,1.8 ,2   ,0,0,-0.25,0]), 19882
    else:
        import datetime as dtm
        startyear, startquarter = D_Setup["start_year"], D_Setup["start_quarter"]
        startdate = dtm.datetime(
            startyear,
            int(startquarter*3),
            30+1*(startquarter==1 or startquarter==4)
        )
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
        V_ParSrt, L_ParNam = D_Setup["V_ParSrt"], D_Setup["L_ParNam"]
        V_StaSrt[12] =   TP10  - V_ParSrt[L_ParNam.index('ATP10')] - V_ParSrt[L_ParNam.index('betaTP10U')]*V_StaSrt[0]
        V_StaSrt[13] =   TP2   - V_ParSrt[L_ParNam.index('ATP2')]  - V_ParSrt[L_ParNam.index('BTP2U')]    *V_StaSrt[0] - V_ParSrt[L_ParNam.index('betaTP10TP2')]*TP10
    toPrint = {key:value for key,value in zip(["UGAP","G","Z","PI","FFR","lagUGAP","lagG","lagZ","lagPI","lagFFR","EpsCPI","EpsPRI","EpsTP10","EpsTP2"],V_StaSrt)}
    print(pd.Series(toPrint))
    print("Init_GDP",Init_GDP)
    return V_StaSrt, Init_GDP

def F_BeltonMatrices(p):
    """
    Initialize matrices and update parameters for linear version of Belton et al state space model (ZLB will be tackled with extra code in transition equation)
    This function to create the model matrices looks like REALLY ugly code but is easy to visually inspect (and fast, but this is not a performance critical part of the code). \
    The state space representation is:
    y_t = V_ConObs + M_Design X_t
    X_t = V_ConSta + M_Transi X_{t-1} + M_Select M_CovSho N(0,1)
    The input p is parameters V_ParSrt
    """

    if xp != np:
        if cp.get_array_module(p) == cp:
            p = p.get()  # Make sure params are Numpy

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

def F_SimSta(n_period, n_simula, V_StaSrt, ModelMats, ELB=.125):
    """
    This function is the transition equation, creating paths for the 14 state variables from generated normal shocks (and the initial states)
    """
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
        xp.clip(A_SimShock2States[t,4,:], ELB, None, out = A_SimShock2States[t,4,:])  # Effective Lower Bound at 0.125 %
    return A_SimShock2States

def F_SimObs(A_SimSta, ModelMats, CBO_projection, CBO_weight):
    """
    F_SimObs is the observation equation, mapping the states to 7 observables (UGAP,PI,FFR,PRI,CPI,TP10,TP2). Notice the first three observables are also states.
    """
    n_period, n_simula  = A_SimSta.shape[0], A_SimSta.shape[2]
    M_Design = ModelMats['M_Design']
    A_SimObs = xp.empty((n_period,M_Design.shape[0],n_simula), dtype=xp.float32)
    A_ConObs = xp.swapaxes(xp.tile(ModelMats['V_ConObs'], (n_simula,1)),0,1)
    for t in range(0,n_period): # Shocks are used to compute states, then immediately overwritten with computed states to save memory.
        A_SimObs[t,:,:] = A_ConObs + M_Design @ A_SimSta[t,:,:] # Observation equation: would work a little faster outside loop with "A_SimObs = A_ConObs + (ModelMats['M_Design'] @ A_SimShock2States)",  but uses more memory, which is critical with GPU.
        CBOaddon =  (CBO_projection[min(t//4, len(CBO_projection)-1)]*(1-(t%4)/4) + CBO_projection[min(1+t//4, len(CBO_projection)-1)]*(t%4)/4 ) - xp.mean(A_SimObs[t,6,:], axis=0, keepdims=True)
        A_SimObs[t,6,:] +=  CBO_weight*CBOaddon  * ((t<41) + (t>40)*(t<61)*(1-(t-40)/20))
    return A_SimObs

#Rates block
def MakeFFPaths3(A_SimSta, ModelMats, A_FFPaths, A_CPIPaths, n_exp_horizon=201, ELB=.125):
    """
    MakeFFPaths3 performs a deterministic, 200-quarters-long secondary simulation for every time step of every primary simulations in A_SimSta, to get expectations of future FFR and CPI.
    It stores the average expected future FFR and CPI for horizons up to 200 and 120 quarters, respectively.
    The storage spaces used are A_FFPaths and A_CPIPaths.
    """

    A_SimStaCut = xp.copy(A_SimSta[:,0:11,:])                   # Cut out PRI and TP AR1 states (if passed) to focus on Macro Block core eleven states. Make local copy of to keep overwriting it in loop
    M_Transi = ModelMats['M_Transi'][0:11,0:11]                 # Focus on Macro Block core states 
    A_ConSta = xp.swapaxes(xp.tile(ModelMats['V_ConSta'][0:11], (A_SimStaCut.shape[0],A_SimStaCut.shape[2],1)),1, 2) #Pre-bradcast to shape of A_SimStaCut
    A_FFPaths[:,0, :] = A_SimStaCut[:,4, :]
    A_CPIPaths[:,0, :] = A_SimStaCut[:,3, :] + A_SimStaCut[:,10,:]
    for t in trange(1,n_exp_horizon):                                   # Shocks are used to compute states, then immediately overwritten with computed states to save memory.
        A_SimStaCut = xp.matmul(M_Transi, A_SimStaCut, out=A_SimStaCut) # Transition Equation #1) Matrix multiplication in place. 
        A_SimStaCut += A_ConSta                                                    #2) Addition in place 
        xp.clip(A_SimStaCut[:,4,:], ELB, None, out = A_SimStaCut[:,4,:])           #3) Effective Lower Bound at 12.5 bps annualized rate
        A_FFPaths[:,t,:] = (A_FFPaths[:, t-1 ,:]*(t-1) + A_SimStaCut[:,4,:])/t     # Directly compute average FFrate till horizon
        if t < 30*4+1: #Keep CPI array smaller (5y and 10y tenors only needed for TIPS's IRP, tenors out to 30y needed for FRNs)
            A_CPIPaths[:,t,:]=(A_CPIPaths[:,t-1,:]*(t-1) + A_SimStaCut[:,3,:]+A_SimStaCut[:,10,:])/t   # Same for average CPI

def F_addFRP(Storage):
    """
    This function adds the constant fixed term premia to the passed array.
    The premia are fixed to 10bps for 2y point, 26bps for 5y point,  and interpolated/extrapolated elsewhere.
    """

    n_exp_hor = Storage.shape[1]
    Tenor_y = np.array([2, 5])
    FRP_y = np.array([10, 26]) / 100
    Tenor_q = np.arange(0, n_exp_hor * 0.25, 0.25)
    FRP_q = np.interp(Tenor_q, Tenor_y, FRP_y)
    Storage += xp.reshape(xp.asarray(FRP_q), (1, -1, 1))

def F_addLRP(Storage):
    """
    This function adds the constant liquidity risk premia to the passed array.
    The premia are fixed to 45,32,24,19 bps for the 2y,5y,10y,30y points, and interpolated/extrapolated elsewhere.
    """

    n_exp_hor = Storage.shape[1]
    Tenor_y = np.array([2, 5, 10, 30])
    LRP_y = np.array([45, 32, 24, 19]) / 100
    Tenor_q = np.arange(0, n_exp_hor * 0.25, 0.25)
    LRP_q = np.interp(Tenor_q, Tenor_y, LRP_y)
    Storage += xp.reshape(xp.asarray(LRP_q), (1, -1, 1))

#A
def MakeCoeffTP2_TP10(plot_coeff=True):
    """
    This Function creates interpolated coefficients of ACM term premia at all quarters of tenor between tenor = 2y and tenor = 10y, included, against constant, ACMTP2y, ACMTP10y
    In detail:
    First, ACM are downloaded and saved as csv, if not already done.
    Then ACM Term Premia for yearly tenors between 2y and 10y are regressed against a constant and the ACM term premia at horizons 2y and 10y, obtaining coefficient for yearly tenors.
    Finally, coefficients for quarterly tenors are obtained by interpolating between coefficients for yearly tenors with a polinomial
    """

    try:  # If ACM dataset already downloaded in data folder, just read it
        ACM = pd.read_csv('../data/ACM.csv', index_col=0, parse_dates=True)
    except:  # Otherwise download it and also save a copy to csv for future use
        url_ACM = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
        ACM = pd.read_excel(url_ACM, 'ACM Monthly', index_col=0, parse_dates=True)  # Download data
        ACM = ACM.resample('Q').ffill()  # Resample dates to Quarterly end-of-period
        ACM.to_csv('../data/ACM.csv', index=True)
    # Run linear regressions of ACM Term premia at 3,4,5,6,7,8,9 years tenors against intecept and 2 and 10 years tenors.
    import statsmodels.formula.api as smf
    Reg_Coeff = {}; Dep_Variables = ['ACMTP0' + str(x) for x in range(3, 10)]
    for dep_var in Dep_Variables:
        Reg_Coeff[dep_var] = smf.ols(dep_var + '~ ACMTP02 + ACMTP10', data=ACM).fit().params # Store linear regression results (extract coefficients only)
    # Interpolate coefficients for all quarters between 2 and 10 years with n-degree polynomial (order n chosen to exhaust degrees of freedom)
    Tenor_y = np.arange(2, 11)
    order = len(Tenor_y)
    Tenor_q = np.arange(2, 10.25, 0.25)
    Coeff_q = {}; Coeff_y = {}
    for Regressor in ['Intercept', 'ACMTP02', 'ACMTP10']:
        Coeff_y[Regressor] = np.array([1 * (Regressor == 'ACMTP02')] + [Reg_Coeff[dep_var][Regressor] for dep_var in Dep_Variables] + [1 * (Regressor == 'ACMTP10')]) # Unpack estimated coefficients of ACMTP3y,4y,...9y against Intecept, Beta to ACMTP2y, Beta to ACMTP10y and append the coefficients of ACMTP2, ACMTP10 themselves (0,1,0 and 0,0,1).
        Eval_Coeff = np.poly1d(np.polyfit(Tenor_y, Coeff_y[Regressor], order)) # Fit polynomyal toy yearly coefficients, then build evaluator for polynomial
        Coeff_q[Regressor] = Eval_Coeff(Tenor_q) # Evaluate Polynomial
    if plot_coeff == True:
        fig0 = plt.figure(figsize=[14, 6]);
        plt.plot(Tenor_q, Coeff_q['Intercept'], '.',
                 Tenor_y, Coeff_y['Intercept'], '*',
                 Tenor_q, Coeff_q['ACMTP02'], '.',
                 Tenor_y, Coeff_y['ACMTP02'], '*',
                 Tenor_q, Coeff_q['ACMTP10'], '.',
                 Tenor_y, Coeff_y['ACMTP10'], '*'
                )
        plt.legend(('Interpolated Intercept', 'Estimated Intercept', 'Interpolated Beta to ACMTP2', 'Estimated Beta to ACMTP2', 'Interpolated Beta to ACMTP10', 'Estimated Beta to ACMTP10'), loc='best')
        plt.xlabel('ACM Term premium at tenor (years)');
        plt.ylabel('OLS Coefficient value');
        plt.title('Regression of ACM Term premia on ACM TP10, ACM TP2');
    tpInterp = pd.DataFrame(Coeff_q, index=Tenor_q)
    tpInterp.index.names = ['tenor']
    return tpInterp
#B
def AssumeCoeffTP2_TP10(plot_coeff=True):
    """
    AssumeCoeffTP2_TP10 can be used to assume some values of the coefficients mapping  term premia ACM10y and ACM2y to the term premia for tenors from 0y to 50y
    This is not used if the coefficients between 2y and 10y are estimated with the function MakeCoeffTP2_TP10, in which case the coefficients between 0y and 2y and between 10y and 50y are extrapolated with functions Extrap_TP0_TP2 and Extrap_TP10_TP50

    """

    Tenor_y = np.array([2, 3, 5, 7, 10, 20, 30, 50])
    Tenor_q = np.arange(2, 50.25, 0.25)
    Coeff_q = {};
    Coeff_y = {}
    Coeff_y['Intercept'] = np.array([0, 0, 0, 0, 0, 0.5, 0.4064, 0.45])
    Coeff_y['ACMTP02'] = np.array([1, 0.847, 0.514, 0.255, 0, -0.0397, -0.6203, -.65])
    Coeff_y['ACMTP10'] = np.array([0, 0.185, 0.506, 0.746, 1, 1.0234, 1.2366, 1.3])
    for Regressor in ['Intercept', 'ACMTP02', 'ACMTP10']:
        # Coeff_y[Regressor] = np.append(Coeff_y[Regressor], np.poly1d(np.polyfit(Tenor_y[:-1], Coeff_y[Regressor] ,1))(50) )  # Extrapolate to add 50y point to Coeff_y
        Coeff_q[Regressor] = np.interp(Tenor_q, Tenor_y, Coeff_y[Regressor])  # Interpolate to get Coeff_q
    if plot_coeff == True:
        fig0 = plt.figure(figsize=[14, 6]);
        plt.plot(
            Tenor_q, Coeff_q['Intercept'], '.',
            Tenor_y, Coeff_y['Intercept'], '*',
            Tenor_q, Coeff_q['ACMTP02'], '.',
            Tenor_y, Coeff_y['ACMTP02'], '*',
            Tenor_q, Coeff_q['ACMTP10'], '.',
            Tenor_y, Coeff_y['ACMTP10'], '*'
        )
        plt.legend(('Interpolated Intercept', 'Assumed Intercept', 'Interpolated Beta to ACMTP2',
                    'Assumed Beta to ACMTP2', 'Interpolated Beta to ACMTP10', 'Assumed Beta to ACMTP10'), loc='best')
        plt.xlabel('ACM Term premium at tenor (years)');
        plt.ylabel('OLS Coefficient value');
        plt.title('Projection of ACM Term premia on ACM TP10, ACM TP2');  # plt.show(block=False);
    tpInterp = pd.DataFrame(Coeff_q, index=Tenor_q)
    tpInterp.index.names = ['tenor']
    return tpInterp

#C
def Extrap_TP0_TP2_2(A_Storage, TP02series_q):
    """
    This function extrapolates the term premium for the first 8 quarters ahead tenors by tapering the 2y TP to zero.
    Term premia are added to the passed stored rates for the corresponding tenors.
    """

    for qtr in [0,1,2,3,4,5,6,7]:  # Add term premium for first eight quarters: from +0qtrs (now), to +7qtr (1yr 2qtrs).
        # TP linearly increases in qtrs +4,...,+7 from 0 towards TP2(value at +8 qtrs)
        A_Storage[:, qtr, :] += xp.squeeze(TP02series_q) * qtr / 8

#D
def Interp_TP2_TP10_2(A_Storage, TP02series_q, TP10series_q, Coeff_q):
    """
    This function uses to the coefficients estimated with MakeCoeff_TP2_TP10 (or assumed with AssumeCoeff_TP2_TP10 ) to interpolate the term premia between 2y and 10y tenors (or up to 50y point with assumed coefficients)
    Term premia are added to the passed stored rates.
    """

    cut = len(Coeff_q['Intercept']) + 8  # Cutoff will be 41 (10y point) with estimation and 201 (50y) with assumption
    Intercept_q = xp.reshape(xp.asarray(Coeff_q['Intercept'], dtype=xp.float32), (1, -1, 1))
    Beta_TP02_q = xp.reshape(xp.asarray(Coeff_q['ACMTP02'], dtype=xp.float32), (-1, 1))
    Beta_TP10_q = xp.reshape(xp.asarray(Coeff_q['ACMTP10'], dtype=xp.float32), (-1, 1))
    A_Storage[:, 8:cut, :] += Intercept_q  # Add is in place ...
    A_Storage[:, 8:cut, :] += Beta_TP02_q @ TP02series_q  # ... but Matmul is not. suboptimal for memory management.. but improving seems hard. Can divide in blocks if too big.
    A_Storage[:, 8:cut, :] += Beta_TP10_q @ TP10series_q  # Extra: save separately TP05series_q as intermediate result for next function
    TP05series_q = Intercept_q[:, 4 * (5 - 2), :] + Beta_TP02_q[12] * TP02series_q + Beta_TP10_q[12] * TP10series_q
    return TP05series_q

#E
def Extrap_TP10_TP50_2(A_Storage, TP02series_q, TP05series_q, TP10series_q,
                       plot_conv=False, use10y5yslope=True, use_convadj=False, replicateBeltonTP=True):
    """
    This function extrapolates the term premia for tenors between 10y and 50y by duration. Only used when TP out to 50y are not derived from assumed coefficients.
    0) Convexity adjustments a_{t} on ZCBs are interpolated at all ZCB tenors, using given values of convexity adjustmets for some given ZCB tenors
    1) 10y and 2y (or 5y) Term Premia are converted in convexity adjusted. Suppose we are using 2y and rahter than 5y:
       TP2a = TP2 + a_{2};  TP10a = TP2 + a_{2}
    2) The slope between 2y and 10y convexity adjusted term premia is used to extrapolate convexity adjusted term premia by duration out to 50y . Duration is known (=maturity) because we are dealing with ZCB term premia
       TP50a = TP10a + (50-10) [TP10a-TP2a]/[10-2]
    4) The convexity adjustment is subtracted to get the term premia:
       TP50 = TP50a - a_{50}
    5) Term premia are added to the passed stored rates for the relevant tenors.
    6) To replicate Belton et al, the extrapolated term premia need to be shifted by some ad hoc quantities. This is because we are using different convexity adjustments, in turn due to Belton et al intending the convexity adjustments as specified for par rates (rather than ZCBs) and using an approximation to get the duration of par notes before knowing their TPs.
    """

    Tenor_y = np.array([5, 10, 20, 30, 50])
    ConAdj_y = np.array([2.7, 9.9, 33.4, 67.9, 188.6],
                        dtype=np.float32) / 100  # These are adjustments for ZCBs are different from Belton et al. where convexity adjustments were though of as for par rates, whose duration for extrapolation was approximated by the duration without term premium.
    Tenor_q = np.reshape(np.arange(0, 50.25, 0.25, dtype=np.float32), (-1, 1))
    A_Storage[:, 41:, :] += (TP10series_q + ConAdj_y[Tenor_y == 10][0] * use_convadj)  # Intercept of extrapolation: add 10y premium with convexity adjustment...
    blocks = 8
    size = int((201 - 41) / blocks)
    if use10y5yslope == True:
        Slope = (TP10series_q + ConAdj_y[Tenor_y == 10][0] * use_convadj - (TP05series_q + ConAdj_y[Tenor_y == 5][0] * use_convadj)) / (10 - 5)
    else:  # Use instead the 2y-10y slope as initially mentioned in the paper.
        Slope = (TP10series_q + ConAdj_y[Tenor_y == 10][0] * use_convadj - (TP02series_q + 0)) / (10 - 2)
    for block in range(
            blocks):  # Do ten-year or 5-year horizon blocks to save on GPU memory space, at cost of minor speed deterioration.
        start = 41 + size * block
        end = 41 + size * (block + 1)
        A_Storage[:, start:end, :] += (xp.asarray(Tenor_q[start:end]) - 10) @ Slope
    ConAdj_q = np.interp(Tenor_q, Tenor_y, ConAdj_y)
    if plot_conv == True:
        fig1 = plt.figure(figsize=[16 / 1.1, 9 / 1.1]);
        plt.plot(Tenor_q, 100 * ConAdj_q, '.',
                 Tenor_y, 100 * ConAdj_y, '*');
        plt.legend(('Interpolated Conv. Adjustment', 'Given Conv. Adjustment'), loc='best');
        plt.xlabel('Conv. Adjustment at tenor (years)');
        plt.ylabel('basis points');
        plt.title('Interpolation of Convexity Adjustments');
    if use_convadj == True:
        A_Storage[:, 41:, :] -= xp.asarray(np.expand_dims(np.atleast_2d(ConAdj_q[41:]), 0))
    if replicateBeltonTP == True:  # For exact replication, need to shift levels of TP, since we are not using the same convexity adjustments and same coefficients from ACM.
        Adj_Tenor_y = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50])
        Adj_y = np.array([-0.05, -0.02, -0.10, -0.08, -0.08, -0.08, -0.25, -0.45, -0.15])
        Adj_q = xp.asarray(np.interp(Tenor_q, Adj_Tenor_y, Adj_y), dtype=xp.float32)
        A_Storage += xp.reshape(Adj_q, (1, -1, 1))


def MakeCoupRates(A_IRPaths):
    """
    Converts an array of Zero Yield Curves to an array of Par Yield Curves (in place, except for some extra temporary storage for discount factors)
    """
    A_IRPaths += xp.sign(A_IRPaths)*0.0001*(xp.abs(A_IRPaths)<0.0001)  # To avoid numerical issues, Push absolute value of rates away from exact zero by adding or subtracting 1bp/100 = 0.0001%. 
    exponents = xp.reshape(
        xp.arange(0,A_IRPaths.shape[1],dtype=xp.float32),
        tuple([1,-1] + [1 for x in  range(A_IRPaths.ndim-2)])
    )
    DiscFactors = (1/(1+A_IRPaths/400))**exponents
    for T in trange(1,A_IRPaths.shape[1]):
        A_IRPaths[:,T,...] = 400*(1-DiscFactors[:,T,...])/xp.sum(DiscFactors[:,1:T+1,...], axis=1)

def MakeZCBRates(A_NomsRates):
    """
    Converts an array of Par Yield Curves to an array of Zero Yield Curves (in place!)
    """
    A_NomsRates += xp.sign(A_NomsRates)*0.0001*(xp.abs(A_NomsRates)<0.0001)  # To avoid numerical issues, Push absolute value of rates away from exact zero by adding or subtracting 1bp/100 = 0.0001%. 
    A_NomsRates[:, 1, ...] = 1 / (1 + A_NomsRates[:, 1,...] / 400)  # Transform 1st par (=zcb) rate into 1st ZCB discount factor (do this for 0th par rate as well, inconsequential.)
    for T in range(2,A_NomsRates.shape[1]):                  # Get all other ZCB discount factors recursively "stripping the par curve":
        A_NomsRates[:, T, ...] = (1 - (A_NomsRates[:, T, ...] / 400) * xp.sum(A_NomsRates[:, 1:T, ...], axis=1)) / (1 + A_NomsRates[:, T, ...] / 400)
    A_NomsRates[:, 1:, ...] **= xp.reshape(-1 / xp.arange(1, A_NomsRates.shape[1], dtype=xp.float32), tuple([1, -1] + [1 for x in range(A_NomsRates.ndim - 2)]))  # Invert ZCB discount factors to get ZCB rates
    A_NomsRates[:, 1:, ...] -= 1
    A_NomsRates[:, 1:, ...] *= 400


def MakeOnTheRun(A_Storage):
    """
    Subtracts on-the-run adjustments from par rates.
    """

    Tenor_y =     np.array([    0.25,    2,    3,    5,    7,   10,   20,   30,   50])
    OntheRun_y =  np.array([   -0.07,-0.02,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04,-0.04])
    Tenor_q = np.reshape(np.arange(0,50.25,0.25, dtype=np.float32), (-1,1))
    OntheRun_q = xp.asarray(np.interp(Tenor_q, Tenor_y, OntheRun_y), dtype=xp.float32)
    A_Storage += xp.reshape(OntheRun_q, (1, -1, 1))

def MakeTPPaths2(
    A_SimObs, A_Storage,
    plot_coeff=False, plot_conv=False, use10y5yslope=True, use_convadj=True,
    estimate_not_assume=True, replicateBeltonTP=True):
    """
    Combines functions defined above to add TP for all tenors to the passed yields array. The passed array is the expectational component.
    """

    if estimate_not_assume == True:
        # Time consuming unless ACM data already downloaded. Coeffs between 2y and 10y point.
        CoeffTP2_TP10 = MakeCoeffTP2_TP10(plot_coeff=plot_coeff)
    else:
        # For checking results conditional on given coefficients, coeffs out to 50y point.
        CoeffTP2_TP10 = AssumeCoeffTP2_TP10(plot_coeff=plot_coeff)
    TP02series_q = xp.expand_dims(A_SimObs[:, 5, :], 1)
    TP10series_q = xp.expand_dims(A_SimObs[:, 4, :], 1)
    Extrap_TP0_TP2_2(A_Storage, TP02series_q)
    TP05series_q = Interp_TP2_TP10_2(A_Storage, TP02series_q, TP10series_q, CoeffTP2_TP10)
    if estimate_not_assume == True:
        Extrap_TP10_TP50_2(A_Storage,
                           TP02series_q,
                           TP05series_q,
                           TP10series_q,
                           plot_conv=plot_conv,
                           use10y5yslope=use10y5yslope,
                           use_convadj=use_convadj,
                           replicateBeltonTP=replicateBeltonTP)


# Plotter

def F_PlotRates2(M_avgFFPaths, M_IRPaths, Ratestitle = "XX Rates", avgPathtitle = "Avg Exp. XX", Ylim=[(1,4),(1,4),(-1,2)]): # Please provide already averaged across simualtions:  M_IRPaths = xp.mean(A_IRPaths, axis=2)
    qtr_tenors = [4*yr_tenor for yr_tenor in[1,2,3,4,5,7,10,20,30]]
    if M_IRPaths.shape[1]>=201: 
        qtr_tenors  += [50*4]
    if xp != np:
        M_IRPaths, M_avgFFPaths= M_IRPaths.get(), M_avgFFPaths.get()
    Arrays = [x[:,qtr_tenors] for x in [M_IRPaths, M_avgFFPaths, M_IRPaths-M_avgFFPaths]]
    fig3, axes = plt.subplots(nrows= 2, ncols= 3 , sharex=True, sharey='none', figsize=(16,9))
    for s in [0,1,2]:
        axes[0, s].set_prop_cycle(cycler(color=[plt.cm.get_cmap('rainbow')(x/len(qtr_tenors)) for x in range(1,len(qtr_tenors)+1)]))
        lines = axes[0, s].plot(Arrays[0+s]);
        ylims = axes[0, s].set_ylim(Ylim[s])
        title = axes[0, s].set_title([Ratestitle, avgPathtitle, 'Term Premium'][s])
    leg=axes[1,2].legend(iter(lines), [str(int(q/4))+' Yrs Maturity'for q in qtr_tenors  ], ncol=2, loc='upper right')
    for s in [0,1]:
        title = axes[1, s].set_title(str(2+s*8) + 'y Rate Decomposition')
        lines = axes[1, s].plot( Arrays[0][:,1+s*5], 'black', Arrays[1][:,1+s*5], 'gold', Arrays[2][:,1+s*5], 'magenta' )
        ylims = axes[1, s].set_ylim((-1,4))
    leg=axes[1,0].legend(iter(lines),[Ratestitle, avgPathtitle, 'Term Premium'], loc='best'); 
    axes[1,2].set_axis_off();
        
def F_MakeIRP(A_ExpFFR_05_10, A_SimSta, A_CPIPaths, plot_IRPcoeff=True, plot_IRP=True):
    """
    This function computes the Inflation Risk Premia adding an AR1 residual to a linear function the 5y and 10y real rate gaps.
    In turn, the real rate gaps are defined as the difference of the real rate (=exp. FFR- exp. CPI ) minus R*.
    """

    Store_IRP = xp.zeros((A_SimSta.shape[0], 1 + 4 * 30, A_SimSta.shape[2]), dtype=xp.float32)
    # Create AR1 residuals. Start by creating innovations
    if xp != np: IRP5_and_10 = xp.random.standard_normal(size=(A_SimSta.shape[0], 2, A_SimSta.shape[2]),dtype=xp.float32)  # Utterly important for GPU memory to use directly float32 rather than float64 and then convert. 
    else:        IRP5_and_10 = xp.random.standard_normal(size=(A_SimSta.shape[0], 2, A_SimSta.shape[2])).astype(xp.float32)  # Numpy does not support dtype argument in normal creation
    IRP5_and_10 *= 0.25  # Volatility of innovations to IRP5, IRP10

    for t in range(1, A_SimSta.shape[0]):  # Add up the innovations into the AR1 processes
        IRP5_and_10[t, :, :] += 0.7 * IRP5_and_10[t - 1, :, :]
    IRP5_and_10[:, 0, :] += 0.61 - 0.145 * ((A_ExpFFR_05_10[:, 0, :] - A_CPIPaths[:, 5 * 4, :])  - (A_SimSta[:, 1, :] + A_SimSta[:, 2, :]))  # Add to AR1 to get IRP5
    IRP5_and_10[:, 1, :] += 0.61 - 0.245 * ((A_ExpFFR_05_10[:, 1, :] - A_CPIPaths[:, 10 * 4, :]) - (A_SimSta[:, 1, :] + A_SimSta[:, 2, :]))  # Add to AR1 to get IRP10 
    Tenor_y = np.array([    2,    3,    5,    7,   10,   15,   20,   30])
    Consta  = np.array([    0,    0,    0,    0,    0,    0,    0,    0])
    Betas05 = np.array([ 1.02, 1.18,    1, 0.48,    0, 0.06, 0.10, 0.23])
    Betas10 = np.array([-0.49,-0.49,    0, 0.62,    1, 0.87, 0.88, 0.87])
    # Tenor_y = np.array([      2,      3,    5,      7,   10,     20,     30])  #These are alternative betas.
    # Consta  = np.array([   0.08,   0.06,    0,  -0.02,    0, -0.025,  -0.05])
    # Betas05 = np.array([ 1.4833, 1.3781,    1, 0.538 ,    0,-0.3569,-0.3076])
    # Betas10 = np.array([-0.6137,-0.4722,    0, 0.5038,    1, 1.2343, 1.1907])
    Tenor_q = np.arange(0, 30.25, 0.25)
    Consta_q  = np.interp(Tenor_q, Tenor_y, Consta)
    Betas05_q = np.interp(Tenor_q, Tenor_y, Betas05)
    Betas10_q = np.interp(Tenor_q, Tenor_y, Betas10)

    if plot_IRPcoeff == True:
        fig5 = plt.figure(figsize=[16 / 1.1, 9 / 1.1]);
        plt.plot(Tenor_q, Consta_q, '.',
                 Tenor_y, Consta, '*',
                 Tenor_q, Betas05_q, '.',
                 Tenor_y, Betas05, '*',
                 Tenor_q, Betas10_q, '.',
                 Tenor_y, Betas10, '*'
                );
        plt.legend(('Interpolated Constants',
                    'Given Constants',
                    'Interpolated Betas to IRP 5y',
                    'Given Betas to IRP 5y',
                    'Interpolated Betas to IRP 10y',
                    'Given Betas to IRP 10y')
                  );
        plt.title('Inflation Risk Premia Curve from 5y and 10y points')

    Store_IRP += xp.reshape(xp.asarray(Consta_q), (1, -1, 1))
    Store_IRP += xp.atleast_2d(xp.asarray(Betas05_q)).T @ xp.expand_dims(IRP5_and_10[:, 0, :], 1)
    Store_IRP += xp.atleast_2d(xp.asarray(Betas10_q)).T @ xp.expand_dims(IRP5_and_10[:, 1, :], 1)

    if plot_IRP == True: 
        fig6, axes = plt.subplots(nrows= 3, ncols= 2 , sharex=True, sharey='none', figsize=(16,16)); 
        if xp != np: 
            Arrays = [xp.mean(Store_IRP[:,[x*4 for x in [2,3,5,7,10,15,20,30]],:],2).get(), xp.mean(A_ExpFFR_05_10,2).get(), xp.mean(A_SimSta[:,1,:] + A_SimSta[:,2,:],1).get(), xp.mean(A_CPIPaths[:,[4*5,4*10],:],2).get()]
        else:        
            Arrays = [xp.mean(Store_IRP[:,[x*4 for x in [2,3,5,7,10,15,20,30]],:],2)      , xp.mean(A_ExpFFR_05_10,2)      , xp.mean(A_SimSta[:,1,:] + A_SimSta[:,2,:],1)      , xp.mean(A_CPIPaths[:,[4*5,4*10],:],2)      ]
        colors = [plt.cm.get_cmap('rainbow')(x/len(Tenor_y)) for x in range(1,len(Tenor_y)+1)]
        axes[0, 0].set_prop_cycle(cycler(color=colors))
        Legendpieces = [ str(Tenor_y[x])+'y = '+ str(round(Betas05[x],2))+ ' x 5y ' + ('+'+str(round(Betas10[x],2)))[-5:] + ' x 10y + eps_'+str(Tenor_y[x]) for x in range(len(Tenor_y))]; 
        Legendpieces[2]='5y = 0.61 - 0.145 Rgap5'; 
        Legendpieces[4]='10y = 0.61 - 0.245 Rgap10'
        lines = axes[0,0].plot(Arrays[0]); 
        axes[0,0].set_title('Inflation Risk Premia (mean)'); 
        axes[0,0].legend(iter(lines), Legendpieces, loc=0) 
        lines = axes[0,1].plot((Arrays[1]-Arrays[3])-np.expand_dims(Arrays[2],1)); 
        axes[0,1].set_title('R gap = R - Rstar'); 
        axes[0,1].legend(iter(lines), ('5y Rgap, mean', '10y Rgap, mean'), loc='upper right') 
        lines = axes[1,0].plot(Arrays[1]-Arrays[3]); 
        axes[1,0].set_title('R = Exp Nom - Exp Inf'); 
        axes[1,0].legend(iter(lines), ('5y R, mean', '10y R, mean'), loc='upper right') 
        lines = axes[1,1].plot(Arrays[2], 'k'); 
        axes[1,1].set_title('R star (mean)'); 
        axes[1,1].legend(('Rstar = Z state + G state'), loc='best')
        lines = axes[2,0].plot(Arrays[3]); 
        axes[2,0].set_title('Expected inflation rate (avg future CPI, mean)'); 
        axes[2,0].legend(iter(lines), ('5y horizon', '10y horizon'), loc='upper right') 
        lines = axes[2,1].plot(Arrays[1]); 
        axes[2,1].set_title('Expected nominal rate (avg future FFR, mean)'); 
        axes[2,1].legend(iter(lines), ('5y horizon', '10y horizon'), loc='upper right') 
    
    return Store_IRP

# Overall wrapper for rates block
# Wrapper for memory management, does entire rates block and overwrites intermediate results (Fed Fund rates, TPremia, ZCB rates) and only returns rates on Coupon Bonds issued at par. Averages of intermediate results can be shown setting plot_rates=True

def F_SimRat(A_SimSta, A_SimObs, ModelMats, 
             n_exp_horizon=201, 
             plot_rates=True, plot_coeff = True, plot_conv = True, 
             use10y5yslope=True, use_convadj=True, 
             plot_IRPcoeff=True, plot_IRP = True, 
             estimate_not_assume=True, replicateBeltonTP=True): 
    """
    This wraps together all functions of the rates block. The output are arrays with Nominal Rates, FRN Rates, TIPS Rates
    1) Creates the expectational component in A_Storage with MakeFFPaths3. Expected CPI rates are also stored in A_CPIPaths
    2) Creates the FRN rates adding the FRP to the expectational component
    3) Creates the Nominal Rates adding TP to expectational component. A_Storage is now the Zero curve of Nominal Rates (exp. comp +TP)
    4) Creates the Inflation Risk Premia, IRP, with F_MakeIRP, in the array A_StoreIRP = IRP
    5) To get TIPS rates: switch sign and add nominal rates to have:        A_StoreIRP = (ExpNom + TP) - IRP;
       then subtract exp. CPI infl. and add the liquidity risk premia:      A_StoreIRP =  ExpNom - ExpInf + TP - IRP + LRP
                                                                                       =  (ExpNom-ExpINF) + (TP - IRP - FRP ) + FRP + LRP
                                                                                       =  (ExpNom-ExpINF) + (TP - IRP - FRP ) + FRP + LRP
                                                                                       =          R       +       RRP         + FRP + LRP 
                                                                                       = TIPS 
    7) Maps Zero Curves to Par Curves 
    8) Corrections for Nominals Par Curve: -8bps to account for Bills-FFr spread, and Make on the run. 
    """
    
    A_Storage = xp.empty((A_SimSta.shape[0], n_exp_horizon, A_SimSta.shape[2] ), dtype=xp.float32) 
    A_CPIPaths = xp.empty((A_SimSta.shape[0], 30*4+1, A_SimSta.shape[2] ), dtype=xp.float32) 
    print('MakeFFPaths3')
    MakeFFPaths3(A_SimSta, ModelMats, A_Storage, A_CPIPaths)
    M_avgFFPaths = xp.mean(A_Storage, axis=2)           # Small  extra storage for plotting with plot_rates
    A_ExpFFR_05_10 = xp.copy(A_Storage[:,[4*5,4*10],:]) # Again, extra storage for exp. FFR before overwriting them when adding TP.       
    A_FRNRates = xp.zeros((A_SimSta.shape[0], 1+5*4, A_SimSta.shape[2] ), dtype=xp.float32)  #Create storage for FRN rates
    A_FRNRates += xp.expand_dims(A_Storage[:,1,:],1); 
    F_addFRP(A_FRNRates) #Set FRN rate equal to 3 month expected FFR plus FRP premium.
    print('MakeTPPaths2')
    MakeTPPaths2(A_SimObs, A_Storage, plot_coeff = plot_coeff, 
                 plot_conv = plot_conv, use10y5yslope=use10y5yslope, 
                 use_convadj=use_convadj, estimate_not_assume=estimate_not_assume, 
                 replicateBeltonTP=replicateBeltonTP
                ) #Notice output is Zero rates. It does NOT transform to par coupon rates
    if plot_rates==True :
        F_PlotRates2(M_avgFFPaths, xp.mean(A_Storage, axis=2), Ratestitle = "Noms ZCB Rates", avgPathtitle = "Avg Exp. Fed Funds")
    #Get inflation risk premium for horizons up to 30y: Store = IRP
    print('Make IRP')
    A_StoreIRP = F_MakeIRP(A_ExpFFR_05_10, A_SimSta, A_CPIPaths, plot_IRPcoeff=plot_IRPcoeff, plot_IRP = plot_IRP) 
    A_StoreIRP *= -1                      # Flip sign:                  Store  =    - IRP
    A_StoreIRP += A_Storage[:,:30*4+1,:]  # Add nominal yields:         Store =  (ExpNom + TP) - IRP
    A_StoreIRP -= A_CPIPaths[:,:30*4+1,:] # Subtract exp. inflation...  Store =   ExpNom - ExpINF + TP - IRP
    F_addLRP(A_StoreIRP)                  # ... and Add LRP to get TIPS yield:
    # Store = ExpNom - ExpInf + TP - IRP + LRP  = (ExpNom-ExpINF) + (TP - IRP - FRP ) + FRP + LRP =  R + RRP + FRP + LRP = TIPS 
    if plot_rates==True :
        F_PlotRates2(M_avgFFPaths[:,:30*4+1]-xp.mean(A_CPIPaths[:,:30*4+1,:], axis=2), xp.mean(A_StoreIRP, axis=2), Ratestitle = "Tips ZCB Rates", avgPathtitle = "Avg Exp. (Fed Funds - CPI Infl)")
    # Transform Zero Rates to Par rates
    MakeCoupRates(A_Storage)
    MakeCoupRates(A_StoreIRP)    
    A_Storage[:, 0:5, :] -= 0.08  # Adjust par rates for Bill-FFR basis, 8bps in Belton et al. 
    MakeOnTheRun(A_Storage)       # Adjust par rates for on-the run / off-the-run    
    if plot_rates==True :
        F_PlotRates2(M_avgFFPaths, xp.mean(A_Storage, axis=2), Ratestitle = "Noms Par Rates", avgPathtitle = "Avg Exp. Fed Funds")
        F_PlotRates2(M_avgFFPaths[:,:30*4+1]-xp.mean(A_CPIPaths[:,:30*4+1,:], axis=2), xp.mean(A_StoreIRP, axis=2), Ratestitle = "Tips Par Rates", avgPathtitle = "Avg Exp. (Fed Funds - CPI Infl)")    
    return A_Storage, A_StoreIRP, A_FRNRates 


# Supply effects
def F_Conversion_TYE(A_Rates):
    # Compute duration for par bonds from par rate
    A_Rates += xp.sign(A_Rates)*0.0001*(xp.abs(A_Rates)<0.0001)   # Push absolute value of rates away from exact zero by adding or subtracting 1bp/100 = 0.0001%. 
    exponents = xp.reshape( xp.arange(0,A_Rates.shape[1],dtype=xp.float32),
                            tuple([1,-1] + [1 for x in  range(A_Rates.ndim-2)]))    
    A_Durs = (1/(A_Rates/100)) * (1 - (1 + (A_Rates / 400))**(-exponents) )
    # Conversion of durations to ratio to 10 year durations
    A_10yDur = xp.copy(A_Durs[:,[4*10],:])
    A_Durs /= A_10yDur #Durs are now ratios to 10y
    return A_Durs, A_10yDur
 
def F_TipsNomsBetas(A_NomsRates, A_TipsRates): #Regressions with no constant, ok with differenced data. Betas averaged across simulations.
    X = xp.diff(A_NomsRates[:,40:41,:], axis = 0)                  # Regress on changes of 10y nominal yield
    NomsBetasToNom10y= xp.mean( (xp.sum(X**2,0)**(-1)) * xp.sum(X*xp.diff(A_NomsRates, axis = 0) ,0), axis=1)
    TipsBetasToNom10y= xp.mean( (xp.sum(X**2,0)**(-1)) * xp.sum(X*xp.diff(A_TipsRates, axis = 0) ,0), axis=1)
    X = xp.diff(A_NomsRates[:,:A_TipsRates.shape[1],:], axis = 0)  # Regress on changes of same-maturity nominal yield
    TipsBetasToNoms  = xp.mean( (xp.sum(X**2,0)**(-1)) * xp.sum(X*xp.diff(A_TipsRates, axis = 0), 0), axis=1) 
    return NomsBetasToNom10y, TipsBetasToNom10y, TipsBetasToNoms    


# Debt block
def F_InitiProfiles(
    startyear,
    startquarter,
    No_TIPS_FRN,
    n_exp_horizon, path_CRSP = None, 
    plotFigs=True):
    """
    Given an arbitrary date, this function gets the profile of outstanding face value and coupon rates for TIPS and Nominals at all tenors
    """

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
        FVALUESq.to_csv('../data/FVALUESq.csv', index = True)

        # Compute weighted average coupon rate profile
        COURATEm = CRSP[['MCALDT','MNTSTOMAT','TMTOTOUT_noTIPS','TMTOTOUT_ofTIPS','TCOUPRT']] #'TMYLD' has strictly worse coverage than 'TMPCYLD', and substantially same info on YTM.
        COURATEm = COURATEm.assign(ANNCOUSIZE_noTIPS = COURATEm['TCOUPRT']*0.01*COURATEm['TMTOTOUT_noTIPS'], ANNCOUSIZE_ofTIPS = COURATEm['TCOUPRT']*0.01*COURATEm['TMTOTOUT_ofTIPS'], QTRSTOMAT = np.ceil(COURATEm['MNTSTOMAT']/3)).drop(columns=['TCOUPRT', 'MNTSTOMAT'] )
        COURATEm = COURATEm.fillna(np.inf).groupby(by=['MCALDT', 'QTRSTOMAT']).sum().replace(np.inf, np.nan) 
        COURATEm = COURATEm.assign(AVGCOURT_noTIPS = 100*COURATEm['ANNCOUSIZE_noTIPS']/COURATEm['TMTOTOUT_noTIPS'], AVGCOURT_ofTIPS = 100*COURATEm['ANNCOUSIZE_ofTIPS']/COURATEm['TMTOTOUT_ofTIPS']).drop(columns=['TMTOTOUT_noTIPS','ANNCOUSIZE_noTIPS','TMTOTOUT_ofTIPS','ANNCOUSIZE_ofTIPS'])
        COURATEq = COURATEm.unstack('QTRSTOMAT', fill_value = np.nan).resample('Q').ffill()                 # Horizon to maturity used to create columns. 
        COURATEq = COURATEq.fillna(np.inf).groupby(lambda x: x[0] + str(np.int(x[1])) , axis=1).sum().replace(np.inf, np.nan) #Consolidate 2 levels of labels to column titles
        COURATEq['AVGCOURT_noTIPS0'] = np.NaN; COURATEq['AVGCOURT_ofTIPS0'] = np.NaN
        COURATEq.to_csv('../data/COURATEq.csv', index = True)
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
    if plotFigs == True:
        Init_AvgCoupRate_toplot = np.copy(Init_AvgCoupRate); 
        Init_AvgCoupRate_toplot[np.isnan(Init_AvgCoupRate_toplot)]=-99
        Ticks= [str(round(x/4)) + 'y' for x in np.arange(0,n_exp_horizon,20)] 
        fig, ax = plt.subplots(2,2, sharex=True, figsize=(14,12))
        for col in range(2):
            ax[0,col].plot(Init_DbtFVout[:,col], c='red'); 
            ax[0,col].set_ylabel('Bn USD'); 
            ax[0,col].set_yscale('linear'); 
            ax[0,col].legend(['Face Value Outstanding by tenor']); 
            ax[0,col].set_title('Initial '+ ['Nominal', 'TIPS'][col] +' Debt Profile at start date ' + startdate)
            ax[1,col].plot(Init_AvgCoupRate_toplot[:,col], c='black'); 
            ax1r = ax[1,col].twinx(); 
            ax1r.plot(Init_DbtFVout[:,col], '--',c='red',linewidth=0.75); 
            ax1r.set_ylabel('Bn USD'); 
            ax1r.set_yscale('log'); 
            ax1r.legend(['Log Face Value Outstanding'], loc='lower right'); 
            ax[1,col].set_xticks([x for x in np.arange(0,n_exp_horizon,20)]); 
            ax[1,col].set_xticklabels(Ticks); ax[1,col].set_ylabel('%'); 
            ax[1,col].legend(['Average Coupon Rate on ' + ['Nominal', 'TIPS'][col] + ' Debt by tenor']); 
            ax[1,col].set_title('Initial Average ' + ['Coupon', 'TIPS'][col] + ' Rate Profile at start date ' + startdate)
            ax[1,col].set_xlim((0,4*40)); 
            ax[1,col].set_ylim(0, max(Init_AvgCoupRate_toplot[:,col])+1); 
            ax[0,col].set_ylim(0); 
            ax1r.set_ylim(1,max(Init_DbtFVout[:,col])*50 ); 
            ax[1,col].set_xlabel('Tenor: years ahead')
    Init_TipsFVadj = Init_DbtFVout[:,1]; Init_FrnsFV =   xp.zeros(n_exp_horizon) # Temporary, till we retrieve data on CPI index ratios and FRNs securities from MSPD
    if No_TIPS_FRN== True:
        Init_DbtFVout[:,1]*=0; Init_AvgCoupRate[:,1]*=0; Init_TipsFVadj*=0; Init_FrnsFV*=0 #Drop TIPS and FRNs initial securities.
    return (
        xp.asarray(Init_DbtFVout, dtype=xp.float32),    # Face Values of Nominals and TIPS
        xp.asarray(Init_AvgCoupRate, dtype=xp.float32), # Average Coupon Rates of Nominals and TIPS
        xp.asarray(Init_TipsFVadj, dtype=xp.float32),   # Adjusted Face Values of TIPS
        xp.asarray(Init_FrnsFV, dtype=xp.float32)       # Face Values of FRNs
    )

def F_InitiProfilesMSPD(
    startyear,
    startquarter,
    No_TIPS_FRN,
    n_exp_horizon, path_MSPD = None, 
    plotFigs=True):
    
    """
    Given an arbitrary date, this function gets the profile of outstanding face value and coupon rates for TIPS, FRNs, and Nominal securities at all tenors
    """
    
    try:    # If MSPD dadaset has already been prepared, just read results
        FVmspd_q = pd.read_csv('../data/FVmspd_q.csv', index_col=0, parse_dates=True)
        CPRATEmspd_q = pd.read_csv('../data/CPRATEmspd_q.csv', index_col=0, parse_dates=True)
    except: # Otherwise do the work to prepare the data and then save a copy for next use. 
        try: 
            MKTSEC = pd.read_csv('../data/MSPD_MktSecty_20010131_20211031.csv', index_col=0, parse_dates=True)
        except:
            print("Missing MSPD excel file, download it in data folder as csv from: \n" +
                  "https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/detail-of-marketable-treasury-securities-outstanding \n" +
                  "scroll to `Preview and Download'and select csv, all available dates")
        else:
            # A) Clean Dataset
            # Keep only individual securities recorded by CUSIPS whose string length is either 9 or 7. This drops:
            # 1) Rows with 'Security Class 2 Description' denoting subtotals, like 'Total Unmatured Treasury Bonds'
            # 2) Rows with 'Security Class 2 Description' non answerable, which happens when 'Security Class 1 Description' is either 'Total Marketable' or 'Federal Financing Bank'
            MKTSEC = MKTSEC[MKTSEC['Security Class 2 Description'].str.len()<10  ] 

            # Fill non-answerable values of total outstanding using issued amounts, inflation adjustment amound and redempted amouns when possible. 
            FloatColumns = ['Interest Rate', 'Yield', 'Issued Amount (in Millions)', 'Amount Adjusted for Inflation (in Millions)', 'Redeemed Amount (in Millions)', 'Outstanding Amount (in Millions)']
            #Find values that are not floats... and replace them with NaNs. There was only one instance. 
            for col in FloatColumns:
                legit_nans = MKTSEC[col].isna()
                temp = pd.to_numeric(MKTSEC[col], errors='coerce')
                #type_errors = temp.isna() & (~legit_nans)
                MKTSEC[col] = temp

            # Drop any remaining NaNs (but there should be none left!)
            MKTSEC =  MKTSEC[~ MKTSEC['Outstanding Amount (in Millions)'].isna() ] 
            legit_nans = MKTSEC['Maturity Date'].isna()
            temp = pd.to_datetime(MKTSEC['Maturity Date'], errors='coerce')
            #type_errors = temp.isna() & (~legit_nans)
            MKTSEC['Maturity Date'] = temp

            # Compute Monts to maturity. 
            MKTSEC['curMnt'] = MKTSEC['Calendar Year']*12 +MKTSEC['Calendar Month Number']
            MKTSEC['endMnt'] = MKTSEC['Maturity Date'].dt.year*12+MKTSEC['Maturity Date'].dt.month
            MKTSEC['MNTSTOMAT'] = MKTSEC['endMnt']-MKTSEC['curMnt'] 
            MKTSEC  = MKTSEC[MKTSEC['MNTSTOMAT'] > -1] # Drop Matured Securities (there are very few, namely 9)

            # Compute outstanding amounts by different types of securities. 
            MKTSEC['OUT_TIPS'] = MKTSEC['Outstanding Amount (in Millions)'] * MKTSEC['Security Class 1 Description'].isin(['Inflation-Protected Securities','Inflation-Indexed Notes', 'Inflation-Indexed Bonds'])
            MKTSEC['OUT_Nomi'] = MKTSEC['Outstanding Amount (in Millions)'] * MKTSEC['Security Class 1 Description'].isin(['Bills Maturity Value', 'Notes', 'Bonds'])
            MKTSEC['OUT_FRNs'] = MKTSEC['Outstanding Amount (in Millions)'] * MKTSEC['Security Class 1 Description'].isin(['Floating Rate Notes'])
            MKTSEC['OUT_TIPSadj'] = MKTSEC['OUT_TIPS'] + MKTSEC['Amount Adjusted for Inflation (in Millions)'].fillna(0) * MKTSEC['Security Class 1 Description'].isin(['Inflation-Protected Securities','Inflation-Indexed Notes', 'Inflation-Indexed Bonds'])

            #Search for securities where we do not know the coupon rate (all are Bills and Floating rate Notes), use yield to maturity.  
            MissingCoupRATE = MKTSEC['Interest Rate'].isna() 
            MissingYTM = MKTSEC['Yield'].isna() 
            MKTSEC.loc[MissingCoupRATE, 'Interest Rate'] = MKTSEC.loc[MissingCoupRATE, 'Yield'] 

            FVmspd_m = MKTSEC[['MNTSTOMAT', 'OUT_TIPS', 'OUT_Nomi', 'OUT_FRNs','OUT_TIPSadj']].fillna(np.inf).groupby(by=['Record Date','MNTSTOMAT']).sum().replace(np.inf, np.nan)
            FVmspd_m = FVmspd_m.unstack('MNTSTOMAT', fill_value = 0).resample('M').ffill()                                                                # Horizon to maturity used to create columns. 
            FVmspd_m[[('OUT_TIPS',0), ('OUT_Nomi',0), ('OUT_FRNs',0), ('OUT_TIPSadj',0)]] = FVmspd_m[[('OUT_TIPS',1), ('OUT_Nomi',1), ('OUT_FRNs',1), ('OUT_TIPSadj',1)]].shift(1)              # Assume FV repaid in month just ended was the outstanding for that month measured at the end of previous month.
            FVmspd_q= FVmspd_m.fillna(np.inf).groupby(lambda x: 'FV' + x[0][4:] + str(np.int(np.ceil(x[1]/3))) , axis=1).sum().replace(np.inf, np.nan)   # Aggregate horizons to quarters. Note how 0th columns are unchanged.
            FVmspd_q[['FVTIPS0','FVNomi0','FVFRNs0', 'FVTIPSadj0']] = FVmspd_q[['FVTIPS0','FVNomi0','FVFRNs0','FVTIPSadj0']].rolling(3).sum(skipna=False);                           # Aggregate FV repaid in last 3 months to get FV repaid in last quarter. 
            FVmspd_q = FVmspd_q.resample('Q').ffill() 
            FVmspd_q.to_csv('../data/FVmspd_q.csv', index = True)

            # Compute weighted average coupon rate profile
            CPRATEmspd_m = MKTSEC[['MNTSTOMAT', 'OUT_TIPSadj', 'OUT_Nomi', 'Interest Rate']] 
            CPRATEmspd_m = CPRATEmspd_m.assign(ANNCOUSIZE_noTIPS = CPRATEmspd_m['Interest Rate']*0.01*CPRATEmspd_m['OUT_Nomi'], ANNCOUSIZE_ofTIPS = CPRATEmspd_m['Interest Rate']*0.01*CPRATEmspd_m['OUT_TIPSadj'], QTRSTOMAT = np.ceil(CPRATEmspd_m['MNTSTOMAT']/3)).drop(columns=['Interest Rate', 'MNTSTOMAT'] )
            CPRATEmspd_m = CPRATEmspd_m.fillna(np.inf).groupby(by=['Record Date', 'QTRSTOMAT']).sum().replace(np.inf, np.nan) 
            CPRATEmspd_m = CPRATEmspd_m.assign(AVGCOURT_noTIPS = 100*CPRATEmspd_m['ANNCOUSIZE_noTIPS']/CPRATEmspd_m['OUT_Nomi'], AVGCOURT_ofTIPS = 100*CPRATEmspd_m['ANNCOUSIZE_ofTIPS']/CPRATEmspd_m['OUT_TIPSadj']).drop(columns=['OUT_Nomi','ANNCOUSIZE_noTIPS','OUT_TIPSadj','ANNCOUSIZE_ofTIPS'])
            CPRATEmspd_q = CPRATEmspd_m.unstack('QTRSTOMAT', fill_value = np.nan).resample('Q').ffill()                 # Horizon to maturity used to create columns. 
            CPRATEmspd_q = CPRATEmspd_q.fillna(np.inf).groupby(lambda x: x[0] + str(np.int(x[1])) , axis=1).sum().replace(np.inf, np.nan) #Consolidate 2 levels of labels to column titles
            CPRATEmspd_q['AVGCOURT_noTIPS0'] = np.NaN; CPRATEmspd_q['AVGCOURT_ofTIPS0'] = np.NaN
            CPRATEmspd_q.to_csv('../data/CPRATEmspd_q.csv', index = True)

    startdate= str(int(startyear)) +'-'+ ('0' + str(int(startquarter*3)))[-2:] +'-'+ str(30+1*(startquarter==1 or startquarter==4))
    MaxAhead = int(FVmspd_q.shape[1]/4); 
    Init_DbtFVout=np.zeros((n_exp_horizon,2));  
    Init_DbtFVout[0:min(MaxAhead,n_exp_horizon),0] = FVmspd_q.loc[startdate, ['FVNomi' + str(x) for x in np.arange(MaxAhead)]]
    Init_DbtFVout[0:min(MaxAhead,n_exp_horizon),1] = FVmspd_q.loc[startdate, ['FVTIPS' + str(x) for x in np.arange(MaxAhead)]]
    Init_DbtFVout /= 1000 ;  #Divide by 1000 to change units from millions to billions.  
    MaxAheadc = int(CPRATEmspd_q.shape[1]/2)
    Init_AvgCoupRate=np.empty((n_exp_horizon,2)); Init_AvgCoupRate[:]= np.nan; 
    Init_AvgCoupRate[0:min(MaxAheadc,n_exp_horizon),0] = CPRATEmspd_q.loc[startdate, ['AVGCOURT_noTIPS' + str(x) for x in np.arange(MaxAheadc)]]
    Init_AvgCoupRate[0:min(MaxAheadc,n_exp_horizon),1] = CPRATEmspd_q.loc[startdate, ['AVGCOURT_ofTIPS' + str(x) for x in np.arange(MaxAheadc)]]

    Init_TipsFVadj =   np.zeros(n_exp_horizon);  
    Init_TipsFVadj[0:min(MaxAhead,n_exp_horizon)] = FVmspd_q.loc[startdate, ['FVTIPSadj' + str(x) for x in np.arange(MaxAhead)]]
    Init_TipsFVadj /= 1000

    Init_FrnsFV =   np.zeros(n_exp_horizon);  
    Init_FrnsFV[0:min(MaxAhead,n_exp_horizon)] = FVmspd_q.loc[startdate, ['FVFRNs' + str(x) for x in np.arange(MaxAhead)]]
    Init_FrnsFV /= 1000

    if plotFigs==True:
        Init_AvgCoupRate_toplot = np.copy(Init_AvgCoupRate); 
        Init_AvgCoupRate_toplot[np.isnan(Init_AvgCoupRate_toplot)]=-99
        Ticks= [str(round(x/4)) + 'y' for x in np.arange(0,n_exp_horizon,20)] 
        fig, ax = plt.subplots(2,2, sharex=True, figsize=(14,12))
        for col in range(2):
            ax[0,col].plot(Init_DbtFVout[:,col], c='red'); ax[0,col].set_ylabel('Bn USD'); ax[0,col].set_yscale('linear'); ax[0,col].legend(['Face Value Outstanding by tenor']); ax[0,col].set_title('Initial '+ ['Nominal', 'TIPS'][col] +' Debt Profile at start date ' + startdate)
            ax[1,col].plot(Init_AvgCoupRate_toplot[:,col], c='black'); 
            ax1r = ax[1,col].twinx(); ax1r.plot(Init_DbtFVout[:,col], '--',c='red',linewidth=0.75); ax1r.set_ylabel('Bn USD'); ax1r.set_yscale('log'); ax1r.legend(['Log Face Value Outstanding'], loc='lower right'); 
            ax[1,col].set_xticks([x for x in np.arange(0,n_exp_horizon,20)]); ax[1,col].set_xticklabels(Ticks); ax[1,col].set_ylabel('%'); ax[1,col].legend(['Average Coupon Rate on ' + ['Nominal', 'TIPS'][col] + ' Debt by tenor']); ax[1,col].set_title('Initial Average ' + ['Coupon', 'TIPS'][col] + ' Rate Profile at start date ' + startdate)
            ax[1,col].set_xlim((0,4*40)); ax[1,col].set_ylim(0, max(Init_AvgCoupRate_toplot[:,col])+1); ax[0,col].set_ylim(0); ax1r.set_ylim(1,max(Init_DbtFVout[:,col])*50 ); ax[1,col].set_xlabel('Tenor: years ahead')
            if col ==1:
                ax[0,col].plot(Init_TipsFVadj, '+', c='blue', label=''); ax[0,col].legend(['Face Value Outstanding by tenor', 'CPI-Adjusted Face Value']), ax[0,col].set_ylim((0,max(Init_TipsFVadj)+1 ))
        fig, ax = plt.subplots(1,1, figsize=(7,6))
        ax.plot(Init_FrnsFV, c='red'); ax.set_ylabel('Bn USD'); ax.set_yscale('linear'); ax.legend(['Face Value Outstanding by tenor']); ax.set_title('Initial FRNs Debt Profile at start date ' + startdate)
        ax.set_ylim(0); ax.set_xticks([x for x in np.arange(0,n_exp_horizon,20)]); ax.set_xticklabels(Ticks)

    if No_TIPS_FRN== True:
        Init_DbtFVout[:,1]*=0; Init_AvgCoupRate[:,1]*=0; Init_TipsFVadj*=0; Init_FrnsFV*=0 #Drop TIPS and FRNs initial securities.
    return (
        xp.asarray(Init_DbtFVout, dtype=xp.float32),    # Face Values of Nominals and TIPS
        xp.asarray(Init_AvgCoupRate, dtype=xp.float32), # Average Coupon Rates of Nominals and TIPS
        xp.asarray(Init_TipsFVadj, dtype=xp.float32),   # Adjusted Face Values of TIPS
        xp.asarray(Init_FrnsFV, dtype=xp.float32)       # Face Values of FRNs
    )

def MakeGDPPaths(Init_GDP, A_SimSta): # Notice that fixing initial price level as base, Init_GDP = Init_RGDP.   
    """
    This function creates paths for GDP from an initial level.
    1) uses the Belton et al 'Okun Law' to map initial real GDP to initial potential GDP through the UGAP.
    2) potential real gdp grows following the G state
    3) Okun law is used again to get RGDP from potential gdp.
    4) PCE is used to map RGDP to NGDP.
    """
    Init_RGDP = Init_GDP  #Just use our initial date as deflator start.
    Init_Pot_RGDP = Init_RGDP / (1 + 2*A_SimSta[0,0,0]/400) # POTRGDP = RGDP / (1 + RGDPGAP%), where RGDPGAP=2 x Unemployment Gap by Okun Law. Take 0th simulation UGAP wlog since in period zero all simulations start at assumed initial state. 
    A_Pot_RGDP = Init_Pot_RGDP * xp.cumprod(1+(A_SimSta[:,1,:])/400, axis=0)  # Potential RGDP quarterly growth is G state.
    A_RGDP = A_Pot_RGDP * (1 + 2*A_SimSta[:,0,:]/400)                         # Apply Okun Law again to get RGDP paths
    A_NGDP = A_RGDP * xp.cumprod(1+(A_SimSta[:,3,:])/400, axis=0)             # Multiply by cumulative inflation to get nominal GDP
    return A_NGDP

def F_MakeDebtStorages(n_period,n_exp_horizon,n_simula): # Prepares the storage spaces for Debt Block
    """
    Prepares storage arrays that will be used in the main debt block function
    """
    
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
    DebtTYE      = xp.copy(A_IRCost)
    del n_period,n_exp_horizon,n_simula
    return locals().copy()

def F_MakeRateStorages(Securities, A_NomsRates, A_TipsRates, A_FrnsRates):
    NomsWhere,  TipsWhere,  FrnsWhere =  [  Securities[0,:]==x  for x in [0,1,2]][:]
    NomsTenors, TipsTenors, FrnsTenors = [4*Securities[1,x].astype(np.int32) for x in [NomsWhere,TipsWhere,FrnsWhere]][:]
    NomsPos,       TipsPos,    FrnsPos = [np.arange(len(x))[x] for x in [NomsWhere,  TipsWhere,  FrnsWhere]][:]
    MaxFrnsTen = np.max(np.append(FrnsTenors,-1))
    A_NomsRates_view, A_TipsRates_view, A_FrnsRates_view = A_NomsRates[:,NomsTenors,:], A_TipsRates[:,TipsTenors,:], A_FrnsRates[:,:MaxFrnsTen+1,:] #Prepare views in advance.
    TYE_ratio_Noms = A_NomsRates[0, :, :]*0
    TYE_ratio_Tips = A_TipsRates[0, :, :]*0
    A_NomsSupEf = A_NomsRates[0, :, :]*0
    A_NomsSupEf_view = A_NomsSupEf[NomsTenors,:]    
    A_TipsSupEf = A_TipsRates[0, :, :]*0
    A_TipsSupEf_view = A_TipsSupEf[TipsTenors,:] 
    return NomsPos,  TipsPos,  FrnsPos, NomsTenors, TipsTenors, FrnsTenors, MaxFrnsTen, A_NomsRates_view, A_TipsRates_view, A_FrnsRates_view, TYE_ratio_Noms, TYE_ratio_Tips, A_NomsSupEf, A_NomsSupEf_view, A_TipsSupEf, A_TipsSupEf_view, A_NomsRates, A_TipsRates

def MakeDbtPaths1(
    Init_DbtFVout,
    Init_AvgCoupRate,
    Init_TipsFVadj,
    Init_FrnsFV,
    IssuanceStrat,
    NomsPos,
    TipsPos,
    FrnsPos,
    NomsTenors,
    TipsTenors,
    FrnsTenors,
    MaxFrnsTen,
    A_NomsRates_view,
    A_TipsRates_view,
    A_FrnsRates_view,
    TYE_ratio_Noms,
    TYE_ratio_Tips,
    A_NomsSupEf,
    A_NomsSupEf_view,
    A_TipsSupEf,
    A_TipsSupEf_view,
    A_NomsRates,
    A_TipsRates,
    A_SimObs,
    A_NGDP,
    A_NomsFV,
    A_TipsFV, 
    A_TipsFVadj, 
    A_TipsFVmax, 
    A_TipsFVmaxOLD, 
    A_FrnsFV,
    A_IRCost, 
    A_TipsFVCost, 
    A_DbtSvc,
    A_TotDfc, 
    Avg_IssRate, 
    A_TotCoup, 
    Store_Pvals, 
    TotDebt, 
    WAM, 
    DebtTYE,
    TrackTYEDebt = False,
    SupplyEffects = False,
    baselineDebtTYE_GDP = None,
    M_Kernels=None,
    CoeffstoConst_and_MEVs = None, 
    UnadjustedKernelIssuance=None, 
    SumUnadjustedKernelIssuance=None,
    TrackWAM=False, 
    Dynamic = False, 
    QuartersperCoup=1,
    n_exp_horizon=201,
    ELB=.125,
    verbose=True): 
    """
    Debt block function.
    1) Initializes the Face Values and Coupons using initial profiles, pre computes quarterly primary deficit and qtly CPI gross inflation
    2) For every quarter t>0
       A) Rolls the storage arrays for face values and coupons by shifting them back 1 position in the the tenor horizons axis, and add a zero for the last horizon.
       B) Updates the inflation adjusted coupons on TIPS by current gross CPI inflation
       C) computes the old an new maximum for tips between face value and inflation adjusted face value
       D) Computes the accrual of TIPS face value adjustment by computing the change in this maximum
       E) computes the cash-flow interest cost suming up coupons generated by nomnal securities and tips securities at all outstanding tenors, adding the interest on FRNs which is equal to their face value times current 3month rate on nominals.
       E.bis) if QuartersperCoup=2, coupons are semiannual: only even quarters ahead generate interest cost (scaled by 2 since coupon rates are kept to be qurterly)
       F) Debt service is interest cost + face value maturing.
       G) Total deficit, or total financing need, is debt service plus primary deficit.
       H) If Dynamic is False, the financing need is split across tenors and securities acording to the issuance shares specified in the passed strategy
       I) If Dynamic is true, the  financing neeed is split across tenors and securities using the pre-computed Kernels x Coefficients x MEVs, plus the part of Kernels x Coefficient x Interest Cost, all adjusted by the h_Index
        the resulting issuance shares are stored for reference.
       L) New issuance is added to face values (at the issued tenors) for all types of securities.
       M) New Issuance times current nominal rates is used to add to the array storing coupon cost generated by outstanding nominal securities. Same for TIPS.
       N) Average issuance rate is computed and stored
       O)Total debt is computed and stored
       P) h_index is updated, if Dynamic=True
       Q) if TrackWAM=True, Weighted Average Maturity of debt is computed and stored.
    After all t are computed, the accrued costs of TIPS face value adjustments are added to the interest costs that generated cash flows.
    """

    n_period = A_NomsRates_view.shape[0]                                          # For clarity
    A_NomsFV[:,:]    = xp.reshape(Init_DbtFVout[:,0],(-1,1))                      # Initial profile of Nominal debt outstanding face values
    A_TipsFV[:,:]    = xp.reshape(Init_DbtFVout[:,1],(-1,1))                      # Initial profile of TIPS debt outstanding face values, NOT INFLATION ADJUSTED
    A_TipsFVadj[:,:] = xp.reshape(Init_TipsFVadj    ,(-1,1))                      # Initial profile of TIPS debt outstanding face values, CPI INFLATION ADJUSTED
    A_FrnsFV[:,:]    = xp.reshape(Init_FrnsFV       ,(-1,1))                      # Initial profile of FRNs debt outstanding face values    
    A_TotCoup[:,0,:] = xp.reshape(Init_DbtFVout[:,0]*xp.nan_to_num(Init_AvgCoupRate[:,0])/400,(-1,1)) # Initial value of coupons on Nominal securities
    A_TotCoup[:,1,:] = xp.reshape(Init_TipsFVadj    *xp.nan_to_num(Init_AvgCoupRate[:,1])/400,(-1,1)) # Initial value of INFLATION ADJUSTED coupons on TIPS securities
    PriDeficit = - (A_SimObs[:, 6, :] / 400) * A_NGDP
    CPIInfl = xp.expand_dims(1 + A_SimObs[:, 3, :] / 400, 0)
    Index = xp.copy(A_IRCost); Index[0,:] = 1; BaseDebt = xp.sum(A_NomsFV[1:,:] + A_TipsFVmax[1:,:] + A_FrnsFV[1:,:], axis=0)
    A_NomsSupEf *= 0 # Make sure initial period supply effect is zero. 
    for t in trange(0,n_period, disable=False if verbose else True):
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
        A_TotCoup[NomsTenors,0,:] +=  (NewIssuance[NomsPos,:]*(A_NomsRates_view[t,:,:]+A_NomsSupEf_view)/400)        
        A_TotCoup[TipsTenors,1,:] +=  (NewIssuance[TipsPos,:]*(A_TipsRates_view[t,:,:]+A_TipsSupEf_view)/400)        # Add new TIPS issuance x TIPS coupon rate to TIPS INFATION ADJUSTED coupons.
        Avg_IssRate[t] = xp.mean(xp.sum((A_NomsRates_view[t,:,:] + A_NomsSupEf_view)* IssuanceStrat[t,NomsPos,:], axis=0) + xp.sum((A_TipsRates_view[t,:,:]+ A_TipsSupEf_view)* IssuanceStrat[t,TipsPos,:], axis=0)  + xp.sum(A_FrnsRates_view[t,FrnsTenors,:] * IssuanceStrat[t,FrnsPos,:], axis=0) )
        Store_Pvals  =  A_NomsFV[1:,:] + A_TipsFVmax[1:,:] + A_FrnsFV[1:,:]
        TotDebt[t,:] = xp.sum(Store_Pvals, axis=0)
        if TrackTYEDebt == True:
            #TYE_ratios = xp.squeeze(F_Conversion_TYE(xp.expand_dims(A_NomsRates[t,:,:]+A_NomsSupEf, axis=0))[0] )
            TYE_ratio_Noms, Nom_10yDur = F_Conversion_TYE(xp.expand_dims(A_NomsRates[t,:,:]+A_NomsSupEf, axis=0))
            TYE_ratio_Tips, Tip_10yDur = F_Conversion_TYE(xp.expand_dims(A_TipsRates[t,:,:]+A_NomsSupEf[:A_TipsRates.shape[1],:], axis=0))
            DebtTYE[t,:] =  xp.sum(A_NomsFV * xp.squeeze(TYE_ratio_Noms), axis=0)
            DebtTYE[t,:] += xp.sum(A_TipsFVmax[:A_TipsRates.shape[1],:] * xp.squeeze((TYE_ratio_Tips*Tip_10yDur)/Nom_10yDur), axis=0)
            if SupplyEffects == True: 
                A_NomsSupEf = 0.06 * xp.squeeze(TYE_ratio_Noms) * xp.expand_dims(100*(DebtTYE[t,:]/A_NGDP[t,:] - baselineDebtTYE_GDP[t,:]), axis=0)
                A_TipsSupEf = 0.06 * xp.squeeze((TYE_ratio_Tips*Tip_10yDur)/Nom_10yDur) * xp.expand_dims(100*(DebtTYE[t,:]/A_NGDP[t,:] - baselineDebtTYE_GDP[t,:]), axis=0)
                A_NomsSupEf = xp.maximum(A_NomsSupEf, -A_NomsRates[min(t+1, n_period-1),:,:]+ELB)  # No supply effect pushing rates to below ELB...
                A_TipsSupEf = xp.maximum(A_TipsSupEf, -A_TipsRates[min(t+1, n_period-1),:,:]+ELB+1) # No supply effect pushing Tips rates to below ELB - 1%...
                A_NomsSupEf_view = A_NomsSupEf[NomsTenors,:] 
                A_TipsSupEf_view = A_TipsSupEf[TipsTenors,:] 
        if ((Dynamic == True) and (t < n_period-1)): 
            Index[t+1,:] = xp.sqrt(TotDebt[t,:] / BaseDebt)   #Debt stock growth index to inflate the issuence amount with Kernels other than Baseline in Dynamic strategies. 
        if TrackWAM == True:
            Store_Pvals  /= xp.expand_dims(TotDebt[t,:], axis=0) #Get distribution of share of PValue of debt across tenors. 
            WAM[t,:]     =  0.25 * np.squeeze(xp.reshape(xp.arange(1,n_exp_horizon,dtype=xp.int32),(1,-1)) @ Store_Pvals)
    A_IRCost += A_TipsFVCost #Add to the cash flow cost the accrued one

def Performance(
    Init_DebtProfiles,
    RateStorages,
    A_SimObs,
    A_NGDP,
    Securities,
    ELB=0.125,
    Const_and_MEVs=None,
    M_Kernels=None,
    CoeffstoConst_and_MEVs=None,
    SingleIssuance = False, 
    Static = False,
    Dynamic = False, 
    QuartersperCoup=1,
    n_period = 80,
    n_exp_horizon = 201,
    n_simula = 2000,
    TrackTYEDebt = False,
    SupplyEffects = False,
    baselineDebtTYE_GDP = None,
    verbose = True):
    """
    This function takes as key input the set of Securities to be issued, and issuance strategy in terms of Kernels and Coefficients of Kernels to MEVs and constant.
    If Static = True, Kernel x Coeff x Constant gives the strategy as issuance shares.
    If Dynamic = True, the Kernel x Coeff x MeVs is pre-computed for all MEVs except interest cost MEV. The component generated by Interest Cost MEV will be added inside the debt block. This will give a dollar amount to be issued at all securities and tenors (to be adjusted by h_index).
    If SingleIssuance = True, the strategies simply set to are include all strategies that set issuance shate to 1 for one of the given securities.
    Then for each strategy, the strategy, interest rates arrays, storage arrays for debt objects, and initial debt profiles are passed to the debt block function
    Finally, performance statistics are computed: average interest cost, std dev of interest cost, std dev of total deficit, and correlation between primary deficit and interest cost. Notice that statistics are computed across simulations, for the final period (or averaged across some periods)
    """

    assert (SingleIssuance + Static + Dynamic == 1), "Set to True one and only one option: SingleIssuance, Stati, Dynamic"
    n_securi = Securities.shape[1]
    if SingleIssuance == True: N_strats = n_securi; CoeffstoConst_and_MEVs_i = None
    else: N_strats = CoeffstoConst_and_MEVs.shape[2]
    Avg_IssRate = xp.zeros(N_strats, dtype=xp.float32)
    Avg_IRCost = xp.zeros(N_strats, dtype=xp.float32)
    Std_IRCost = xp.zeros(N_strats, dtype=xp.float32)
    Std_TotBal = xp.zeros(N_strats, dtype=xp.float32)
    Cor_IRC_PRI = xp.zeros(N_strats, dtype=xp.float32)
    DebtStorages = F_MakeDebtStorages(n_period, n_exp_horizon, n_simula)
    IssuanceStrat = xp.zeros((n_period, n_securi, n_simula), dtype=xp.float32)
    UnadjustedKernelIssuance, SumUnadjustedKernelIssuance = None, None
    for i in trange(0, N_strats, disable=False if verbose else True):
        if SingleIssuance == True:
            IssuanceStrat*=0; IssuanceStrat[:,i,:]=1
        elif Static == True:
            CoeffstoConst_and_MEVs_i = CoeffstoConst_and_MEVs[:,:,i]
            IssuanceStrat[:,:,:] = xp.tile( xp.expand_dims(M_Kernels @ CoeffstoConst_and_MEVs_i, axis=0),  (n_period, 1, n_simula)   )
        elif Dynamic == True: #Dynamic: Kernels coefficients give dollars of issuance, not shares.  IssuanceStrat will be overwritten
            CoeffstoConst_and_MEVs_i = CoeffstoConst_and_MEVs[:,:,i]
            UnadjustedKernelIssuance =   M_Kernels[:,1:]    @  (CoeffstoConst_and_MEVs_i @ Const_and_MEVs) # Dollar Issuance caused by Kernelse else than Baseline   
            SumUnadjustedKernelIssuance = xp.sum(UnadjustedKernelIssuance, axis = 1) 
        MakeDbtPaths1(*Init_DebtProfiles, IssuanceStrat, *RateStorages, A_SimObs, A_NGDP, *list(DebtStorages.values()), TrackTYEDebt, SupplyEffects , baselineDebtTYE_GDP , M_Kernels, CoeffstoConst_and_MEVs_i, UnadjustedKernelIssuance, SumUnadjustedKernelIssuance, ELB=ELB,TrackWAM=False, Dynamic = Dynamic, QuartersperCoup=QuartersperCoup, verbose = False)
        DebtStorages['A_IRCost'] /= A_NGDP
        DebtStorages['A_IRCost'] *= 400
        Axis = 1  # Select 1 to compute statistics across simulations for a fixed period, then average across periods. Select 0 to do the converse: compute stats across periods for a fixed simulation, then average across simulations.
        Startperiod = 79  # Starting period for statistics window. Select a number from 1 to 79 (or -1)
        Avg_IssRate[i] = xp.mean(DebtStorages['Avg_IssRate'][Startperiod:])
        Avg_IRCost[i] = xp.mean(xp.mean(DebtStorages['A_IRCost'][Startperiod:,:], axis= Axis ))
        Std_IRCost[i] = xp.mean(xp.std( DebtStorages['A_IRCost'][Startperiod:,:], axis= Axis )) # xp.mean(xp.std( A_IRCost[1:,:], axis=0 ))
        Std_TotBal[i] = xp.mean(xp.std(-A_SimObs[Startperiod:,6,:] + DebtStorages['A_IRCost'][Startperiod:,:], axis=Axis )) #xp.mean(xp.std(-A_SimObs[1:,6,:] + A_IRCost[1:,:], axis=0 ))
        Cor_IRC_PRI[i] = xp.mean(  xp.mean((DebtStorages['A_IRCost'][Startperiod:,:] - xp.mean(DebtStorages['A_IRCost'][Startperiod:,:], axis=Axis, keepdims=True))*(-A_SimObs[Startperiod:,6,:] - xp.mean(-A_SimObs[Startperiod:,6,:], axis=Axis, keepdims=True)), axis=Axis)/(xp.std(DebtStorages['A_IRCost'][Startperiod:,:], axis=Axis)*xp.std(-A_SimObs[Startperiod:,6,:], axis=Axis))  )
    return Avg_IssRate, Avg_IRCost, Std_IRCost, Std_TotBal, Cor_IRC_PRI

def PlotSims2(A_SimObs, A_SimSta, A_NomsRates, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP, DebtTYE): 
    D_VarsToPlot = {'UGAP':(-2,2), 'Inflation':(0,4),'R Star':(0,2) ,'Fed Funds':(0,7), 'PRI':(-6,3), 'Int. Cost/GDP':(1,4), 'Tot. Balance/GDP':(-8,3), 'At Par Coupon Rates':(1,4),'TP2':(-1,0.5), 'TP10':(-1,1.5), 'G.state':(0,3), 'Rollovers/GDP':(10,30), 'Debt/GDP':(60,120),'TYE Nom. Debt/GDP':(30,80),'WAC':(1,5),'WAM':(5,7) }
    L_VarsToPlot = list(D_VarsToPlot.keys()); axlims = list(D_VarsToPlot.values()) 
    n_VarstoPlot = len(L_VarsToPlot)
    n_TotCol = 4
    n_TotRow = np.ceil(n_VarstoPlot/n_TotCol).astype(int)
    M_CRPaths = xp.mean(A_NomsRates, axis=2)  #Only interested in plotting mean rates, so don't bring all array back from GPU
    if xp != np: # Bring arrays back from GPU, only for the intial state vector that we want to plot
        Arrays = [x.get() for x in [A_SimObs, A_SimSta, M_CRPaths, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP, DebtTYE] ]
    else: 
        Arrays = [x       for x in [A_SimObs, A_SimSta, M_CRPaths, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP, DebtTYE] ]
    A_SimObs, A_SimSta, M_CRPaths, A_IRCost, A_DbtSvc, A_TotDfc, TotDebt, WAM, A_NGDP, DebtTYE = Arrays[:]; del Arrays
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
        elif (VarName == 'Int. Cost/GDP'):      A = 400*A_IRCost/A_NGDP #Annualized: multiply by 4
        elif (VarName == 'Tot. Balance/GDP'):   A =  A_SimObs[:,L_ObsNam.index('PRI'),:] - 400*A_IRCost/A_NGDP
        elif (VarName == 'Rollovers/GDP'):      A = 400*(A_DbtSvc - A_IRCost)/A_NGDP
        elif (VarName == 'Debt/GDP'):           A = 100 * TotDebt/A_NGDP 
        elif (VarName ==  'TYE Nom. Debt/GDP'): A = 100 * DebtTYE/A_NGDP 
        elif (VarName == 'WAM'):      A = WAM
        elif (VarName == 'WAC'):      A = 400*A_IRCost/TotDebt
        if VarName!= 'At Par Coupon Rates': lines = axes[row, col].plot(np.percentile(A, np.array([15,30,50,70,85]), axis=1).T)
        else: qtr_tenors = [4*yr_tenor for yr_tenor in[1,2,3,4,5,7,10,20,30,50]];  axes[row, col].set_prop_cycle(cycler(color=[plt.cm.get_cmap('rainbow')(x/len(qtr_tenors)) for x in range(1,len(qtr_tenors)+1)])); lines = axes[row, col].plot(M_CRPaths[:,qtr_tenors])
        ylims = axes[row, col].set_ylim(axlims[Var])
    plt.legend(iter(lines), ('15%', '30%', 'median', '70%', '85%'), loc='upper right'); #plt.show(block=False)

def PlotStrats(
        Avg_IssRate,
        Avg_IRCost,
        Std_IRCost,
        Std_TotBal,
        Cor_IRC_PRI,
        Title,
        StratNames=[],
        ColorList=None,
        M_style='o',
        Y_lim=(None, None),
        X_lim=(None, None),
        XX_lim=(None, None),
        Tabulate=True):
    """
    Plots the performance frontiers of strategies in the interest cost - risk space, where risk is std (interest cost) for the right panel and std(total deficit) for the right panel
    Also, tabulates the performance statistics and average issuance rates.
    """

    if xp != np:
        X = Std_IRCost.get(); Y = Avg_IRCost.get(); XX = Std_TotBal.get(); XXX = Cor_IRC_PRI.get(); Z = Avg_IssRate.get()
    else:
        X = Std_IRCost; Y = Avg_IRCost; XX = Std_TotBal; XXX = Cor_IRC_PRI; Z = Avg_IssRate
    fig4, axes = plt.subplots(nrows=1, ncols=2, sharex='none', sharey=True, figsize=(16, 4))
    axes[0].scatter(X, Y, c=ColorList, marker=M_style);
    axes[0].set_xlabel("Std IR Cost");
    axes[0].set_ylabel("Avg IR Cost");
    axes[0].set_title(Title + " AvG int cost vs StD int cost", fontsize=9);
    axes[0].set_ylim(Y_lim);
    axes[0].set_xlim(X_lim)
    axes[1].scatter(XX, Y, c=ColorList, marker=M_style);
    axes[1].set_xlabel("Std Deficit");
    axes[1].set_ylabel("Avg IR Cost");
    axes[1].set_title(Title + " AvG int cost vs StD (int cost + pri. deficit)", fontsize=9);
    axes[1].set_xlim(XX_lim)
    for i in range(1):
        if X_lim[i] == None:
            X_lim[i] = ((-1)**(1+i))*np.inf
        if XX_lim[i] == None:
            XX_lim[i] = ((-1)**(1+i))*np.inf  
        if Y_lim[i] == None:
            Y_lim[i] = ((-1)**(1+i))*np.inf 
    for i, label in enumerate(StratNames):
        if (Y[i]>Y_lim[0]) & (Y[i]<Y_lim[1]):
            if (X[i]>X_lim[0]) & (X[i]<X_lim[1]):
                axes[0].annotate(label, (X[i], Y[i]))
            if (XX[i]>XX_lim[0]) & (XX[i]<XX_lim[1]):
                axes[1].annotate(label, (XX[i], Y[i]))
    if Tabulate == True:
        row_names = ["Average Issuance Rate", "Average IR Cost/GDP  ", "StDev IR Cost/GDP  ",
                     "StDev (IR Cost/GDP + PRI Deficit/GDP)  ", "Corr (IRCost/GDP , PRI Deficit/GDP)"]
        import pandas as pd
        return pd.DataFrame([np.round(Z, 2), np.round(Y, 2), np.round(X, 2), np.round(XX, 2), np.round(XXX, 2)],
                            row_names, [name[0:5] for name in StratNames])
    
def F_Loss(Weights_K2_K4,
         M_Kernels, 
         Lambda = 1, 
         Avg_IssRate= xp.zeros(1, dtype=xp.float32),
         Avg_IRCost= xp.zeros(1, dtype=xp.float32),
         Std_IRCost= xp.zeros(1, dtype=xp.float32), 
         Std_TotBal= xp.zeros(1, dtype=xp.float32), 
         Cor_IRC_PRI= xp.zeros(1, dtype=xp.float32),
         Init_DebtProfiles = None, 
         RateStorages = None,
         A_SimObs=None,
         A_NGDP = None, 
         D_Setup = None,
         Risk="Std_IRCost"): 
    """
    Computes the Loss from a given static strategy, by:
    1. Creating the strategy as a mix of kernels, using the given static weights.
    2. Invoking th function 'Performance' to simulate the debt path and evaulate the strategy in terms of interest cost and risk.
    3. Combining cost and risk with the given risk aversion parameter.Risk can be the std of interest cost or the std of total balance. 
    """
    Const = xp.ones((D_Setup["n_period"],1, D_Setup["n_simula"]),dtype=xp.float32)
    #M_Kernels = xp.concatenate(
    #(D_Setup["Kernel1_Baseline"] ,D_Setup["Kernel2_Bills"], D_Setup["Kernel3_Belly"], D_Setup["Kernel4_Bonds"]),
    #axis=1)
    CoeffstoConst = xp.ones((M_Kernels.shape[1],1,1), dtype=xp.float32)
    CoeffstoConst[1:,0,0] = xp.array(Weights_K2_K4)
    Avg_IssRate[:], Avg_IRCost[:], Std_IRCost[:], Std_TotBal[:], Cor_IRC_PRI[:] =  Performance(
        Init_DebtProfiles, 
        RateStorages, 
        A_SimObs, 
        A_NGDP, 
        Securities = D_Setup["Securities"], 
        Const_and_MEVs=Const,  
        M_Kernels=M_Kernels, 
        CoeffstoConst_and_MEVs=CoeffstoConst, 
        Static=True, 
        QuartersperCoup=D_Setup["QuartersperCoup"],
        n_period=D_Setup["n_period"],
        n_exp_horizon=D_Setup["n_exp_horizon"],
        n_simula=D_Setup["n_simula"],
        TrackTYEDebt = False,
        SupplyEffects = False,
        baselineDebtTYE_GDP = None,
        verbose = False,
    )
    if Risk=="Std_IRCost":
        loss = Avg_IRCost + Lambda * Std_IRCost
    else:
        loss = Avg_IRCost + Lambda * Std_TotBal
    if xp != np:
        loss = loss.get()
    return loss

from scipy import optimize as opt



def F_StaticOptim(Init_DebtProfiles, 
         RateStorages,
         A_SimObs,
         A_NGDP, 
         D_Setup,
         M_Kernels,
         bnds = ((-2, 2), (-2, 2), (-2,2)), 
         RA_low=0.0, RA_hig=5, RA_num=11, 
         Risk="Std_IRCost"): 
    
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - bnds[0][0]},
            {'type': 'ineq', 'fun': lambda x: -x[0] + bnds[0][1]},
            {'type': 'ineq', 'fun': lambda x:  x[1] - bnds[1][0]},
            {'type': 'ineq', 'fun': lambda x: -x[1] + bnds[1][1]},
            {'type': 'ineq', 'fun': lambda x:  x[2] - bnds[2][0]},
            {'type': 'ineq', 'fun': lambda x: -x[2] + bnds[2][1]})

    Dict = globals()
    Dict["M_Kernels"] = M_Kernels
    cons_pos = ()
    for s in range(D_Setup["Securities"].shape[1]):
        funcode = "lambda x: np.concatenate((np.ones(1,dtype=xp.float32),np.asarray(x,dtype=xp.float32))) @ M_Kernels[" + str(s) + ",:]"
        if xp != np:
            funcode += ".get()" 
        cons_pos += ({'type': 'ineq', 'fun': eval(funcode,  Dict) },)

    #RA_low=0.0  # 1.55
    #RA_hig=5
    #RA_num=11
    Risk_Aversion = xp.concatenate( (xp.array([min(0,RA_low)]) , xp.round(xp.exp(xp.linspace(xp.log(max(0.05,RA_low)),xp.log(RA_hig),num=RA_num-1)),2)))
    Avg_IssRateZ= xp.zeros(RA_num, dtype=xp.float32)
    Avg_IRCostZ = xp.zeros(RA_num, dtype=xp.float32)
    Std_IRCostZ = xp.zeros(RA_num, dtype=xp.float32)
    Std_TotBalZ = xp.zeros(RA_num, dtype=xp.float32)
    Cor_IRC_PRIZ= xp.zeros(RA_num, dtype=xp.float32)

    for i in trange(RA_num):  
        res = opt.minimize(F_Loss, [0,0,0.01], args=(M_Kernels, Risk_Aversion[i],
        Avg_IssRateZ[i:i+1],
        Avg_IRCostZ[i:i+1],
        Std_IRCostZ[i:i+1], 
        Std_TotBalZ[i:i+1], 
        Cor_IRC_PRIZ[i:i+1],
        Init_DebtProfiles,RateStorages,A_SimObs,A_NGDP,D_Setup,Risk), method='COBYLA',  options={'maxiter':500}, tol=0.001, constraints=cons+cons_pos)
        print("Risk Aversion "+ str(xp.round(Risk_Aversion[i],2)))
        print(res) 
    return Avg_IssRateZ, Avg_IRCostZ, Std_IRCostZ, Std_TotBalZ, Cor_IRC_PRIZ, Risk_Aversion
