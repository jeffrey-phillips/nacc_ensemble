'''
CIS 419/519 project: Using decision tree ensembles to infer the pathological 
    cause of age-related neurodegenerative changes based on clinical assessment

nadfahors: Nicole Chiou, Bram Bruno, & Jeff Phillips

This file contains code for preparing NACC data for analysis, including:
    * synthesis of pathology data to create pathology class outcomes
    * dropping uninformative variables from predictor set
    * identifying and merging/resolving redundant clusters of variables
    * identifying missing data codes and replacing with NaNs as appropriate
    * creating change variables from longitudinal data
    * imputation of missing data
    * categorizing retained variables as interval/ratio, ordinal, or nominal
    * creation of dummy variables for nominal variables
    * standardizing interval/ratio and ordinal variables
    * creating date variables, then converting these to useful ages or intervals
    * quadratic expansion for interval/ratio variables?
    
'''

# Module imports
import pandas as pd
import numpy as np
import datetime

# Read in full dataset. Warning: this is about 340 MB.
fulldf = pd.read_csv('investigator_nacc48.csv')

# List of Uniform Data Set (UDS) values that will serve as potential
# predictors. Those with a "False" next to them will be excluded after data
# preparation; those with a True will be kept.
xvar = pd.read_csv('xvar.csv')

# Variables from the NACC neuropathology table that will be used to group
# individuals by pathology class:
#    1) Alzheimer's disease (AD);
#    2) frontotemporal lobar degeneration due to tauopathy (FTLD-tau)
#    3) frontotemporal lobar degeneration due to TDP-43 (FTLD-TDP)
#    4) Lewy body disease due to alpha synuclein (including Lewy body dementia and Parkinson's disease)
#    5) vascular disease
# Path classes: AD (ABC criteria); FTLD-tau; FTLD-TDP, including ALS; Lewy body disease (are PD patients captured here?); vascular
npvar = pd.DataFrame(np.array(["NPPMIH",0, # Postmortem interval--keep in as a potential confound variable?
    "NPFIX",0,
    "NPFIXX",0,
    "NPWBRWT",0,
    "NPWBRF",0,
    "NACCBRNN",0,
    "NPGRCCA",0,
    "NPGRLA",0,
    "NPGRHA",0,
    "NPGRSNH",0,
    "NPGRLCH",0,
    "NACCAVAS",0,
    "NPTAN",False,
    "NPTANX",False,
    "NPABAN",False,
    "NPABANX",False,
    "NPASAN",False,
    "NPASANX",False,
    "NPTDPAN",False,
    "NPTDPANX",False,
    "NPHISMB",False,
    "NPHISG",False,
    "NPHISSS",False,
    "NPHIST",False,
    "NPHISO",False,
    "NPHISOX",False,
    "NPTHAL",False,# Use for ABC scoring to create ordinal measure of AD change
    "NACCBRAA",False,# Use for ABC scoring to create ordinal measure of AD change
    "NACCNEUR",False,# Use for ABC scoring to create ordinal measure of AD change
    "NPADNC",False,# Use for ABC scoring to create ordinal measure of AD change
    "NACCDIFF",False,
    "NACCVASC",False,# Vasc presence/absence
    "NACCAMY",False,
    "NPLINF",False,
    "NPLAC",False,
    "NPINF",False,# Derived variable summarizing several assessments of infarcts and lacunes
    "NPINF1A",False,
    "NPINF1B",False,
    "NPINF1D",False,
    "NPINF1F",False,
    "NPINF2A",False,
    "NPINF2B",False,
    "NPINF2D",False,
    "NPINF2F",False,
    "NPINF3A",False,
    "NPINF3B",False,
    "NPINF3D",False,
    "NPINF3F",False,
    "NPINF4A",False,
    "NPINF4B",False,
    "NPINF4D",False,
    "NPINF4F",False,
    "NACCINF",False,
    "NPHEM",False,
    "NPHEMO",False,
    "NPHEMO1",False,
    "NPHEMO2",False,
    "NPHEMO3",False,
    "NPMICRO",False,
    "NPOLD",False,
    "NPOLD1",False,
    "NPOLD2",False,
    "NPOLD3",False,
    "NPOLD4",False,
    "NACCMICR",False,# Derived variable for microinfarcts
    "NPOLDD",False,
    "NPOLDD1",False,
    "NPOLDD2",False,
    "NPOLDD3",False,
    "NPOLDD4",False,
    "NACCHEM",False,# Derived variables for microbleeds and hemorrhages
    "NACCARTE",False,
    "NPWMR",False,
    "NPPATH",False,# Other ischemic/vascular pathology
    "NACCNEC",False,
    "NPPATH2",False,
    "NPPATH3",False,
    "NPPATH4",False,
    "NPPATH5",False,
    "NPPATH6",False,
    "NPPATH7",False,
    "NPPATH8",False,
    "NPPATH9",False,
    "NPPATH10",False,
    "NPPATH11",False,
    "NPPATHO",False,
    "NPPATHOX",False,
    "NPART",False,
    "NPOANG",False,
    "NACCLEWY",False,# Note that limbic/transitional and amygdala-predominant are not differentiated
    "NPLBOD",False,# But here they are differentiated!
    "NPNLOSS",False,
    "NPHIPSCL",False,
    "NPSCL",False,
    "NPFTDTAU",False,# FTLD-tau
    "NACCPICK",False,# FTLD-tau
    "NPFTDT2",False,# FTLD-tau
    "NACCCBD",False,# FTLD-tau
    "NACCPROG",False,# FTLD-tau
    "NPFTDT5",False,# FTLD-tau
    "NPFTDT6",False,# FTLD-tau
    "NPFTDT7",False,# FTLD-tau
    "NPFTDT8",False,# This is FTLD-tau but associated with ALS/parkinsonism--wut?
    "NPFTDT9",False,# tangle-dominant disease--is this PART? Maybe exclude cases who have this as only path type.
    "NPFTDT10",False,# FTLD-tau: other 3R+4R tauopathy. What is this if not AD? Maybe exclude. How many cases?
    "NPFRONT",False,# FTLD-tau
    "NPTAU",False,# FTLD-tau
    "NPFTD",False,# FTLD-TDP
    "NPFTDTDP",False,# FTLD-TDP
    "NPALSMND",False,# FTLD-TDP (but exclude FUS and SOD1)
    "NPOFTD",False,
    "NPOFTD1",False,
    "NPOFTD2",False,
    "NPOFTD3",False,
    "NPOFTD4",False,
    "NPOFTD5",False,
    "NPFTDNO",False,
    "NPFTDSPC",False,
    "NPTDPA",False,# In second pass, use anatomical distribution to stage 
    "NPTDPB",False,# In second pass, use anatomical distribution to stage
    "NPTDPC",False,# In second pass, use anatomical distribution to stage
    "NPTDPD",False,# In second pass, use anatomical distribution to stage
    "NPTDPE",False,# In second pass, use anatomical distribution to stage
    "NPPDXA",False,# Exclude?
    "NPPDXB",False,# Exclude
    "NACCPRIO",False,# Exclude
    "NPPDXD",False,# Exclude
    "NPPDXE",False,
    "NPPDXF",False,
    "NPPDXG",False,
    "NPPDXH",False,
    "NPPDXI",False,
    "NPPDXJ",False,
    "NPPDXK",False,
    "NPPDXL",False,
    "NPPDXM",False,
    "NPPDXN",False,
    "NACCDOWN",False,
    "NACCOTHP",False,# Survey for exclusion criteria
    "NACCWRI1",False,# Survey for exclusion criteria
    "NACCWRI2",False,# Survey for exclusion criteria
    "NACCWRI3",False,# Survey for exclusion criteria
    "NACCBNKF",False,
    "NPBNKB",False,
    "NACCFORM",False,
    "NACCPARA",False,
    "NACCCSFP",False,
    "NPBNKF",False,
    "NPFAUT",False,
    "NPFAUT1",False,
    "NPFAUT2",False,
    "NPFAUT3",False,
    "NPFAUT4",False,
    "NACCINT",False,
    "NPNIT",False,
    "NPCERAD",False,# What sort of variable?
    "NPADRDA",False,
    "NPOCRIT",False,
    "NPVOTH",False,
    "NPLEWYCS",False,
    "NPGENE",True,# Family history--include in predictors?
    "NPFHSPEC",False,# Code as dummy variables if useful.
    "NPCHROM",False,# Exclusion factor? Genetic/chromosomal abnormalities
    "NPPNORM",False,# Check all the following variables for redundancy with the ones above.
    "NPCNORM",False,
    "NPPADP",False,
    "NPCADP",False,
    "NPPAD",False,
    "NPCAD",False,
    "NPPLEWY",False,
    "NPCLEWY",False,
    "NPPVASC",False,
    "NPCVASC",False,
    "NPPFTLD",False,
    "NPCFTLD",False,
    "NPPHIPP",False,
    "NPCHIPP",False,
    "NPPPRION",False,
    "NPCPRION",False,
    "NPPOTH1",False,
    "NPCOTH1",False,
    "NPOTH1X",False,
    "NPPOTH2",False,
    "NPCOTH2",False,
    "NPOTH2X",False,
    "NPPOTH3",False,
    "NPCOTH3",False,
    "NPOTH3X",0]).reshape((-1,2)))
npvar.columns = ['Variable','Keep']

## Case selection process.

# Include only those with autopsy data.
aut = fulldf[fulldf.NACCAUTP == 1]
del fulldf
def table(a,b):
    print(pd.crosstab(aut[a],aut[b],dropna=False,margins=True))
    
# Exclude for Down's, Huntington's, and other conditions.
aut = aut.loc[aut.DOWNS != 1]
aut = aut.loc[aut.HUNT != 1]
aut = aut.loc[aut.PRION != 1]
aut = aut.loc[~aut.MSAIF.isin([1,2,3])]
aut = aut.loc[~aut.NEOPIF.isin([1,2,3])]
aut = aut.loc[~aut.SCHIZOIF.isin([1,2,3])]
aut.index = list(range(aut.shape[0]))

# How many unique IDs?
# For now, keep in follow-up visits to increase our training data.
uids = aut.NACCID[~aut.NACCID.duplicated()]
#aut = aut[~aut.NACCID.duplicated()]

## Coding of pathology class outcomes.
# Create binary variables for the presence of each pathology class of interest.

# Code Alzheimer's disease pathology based on NPADNC, which implements
# ABC scoring based on Montine et al. (2012).
aut = aut.assign(ADPath = 0)
aut.loc[aut.NPADNC.isin((2,3)),'ADPath'] = 1
aut.loc[aut.NPPAD == 1,'ADPath'] = 1
# The following two commands make the ADPath variable false if the AD path
# diagnosis is as contributing, not as primary.
aut.loc[aut.NPPAD == 2,'ADPath'] = 0
aut.loc[aut.NPCAD == 1,'ADPath'] = 0
aut.loc[aut.NPPVASC == 1,'ADPath'] = 0
aut.loc[aut.NPPLEWY == 1,'ADPath'] = 0
aut.loc[aut.NPPFTLD == 1,'ADPath'] = 0

# Several variables pertain to FTLD tauopathies.
aut = aut.assign(TauPath = [0 for i in range(aut.shape[0])])
aut.loc[aut.NPFTDTAU == 1,'TauPath'] = 1
aut.loc[aut.NACCPICK == 1,'TauPath'] = 1
aut.loc[aut.NACCCBD == 1,'TauPath'] = 1
aut.loc[aut.NACCPROG == 1,'TauPath'] = 1
aut.loc[aut.NPFTDT2 == 1,'TauPath'] = 1
aut.loc[aut.NPFTDT5 == 1,'TauPath'] = 1
aut.loc[aut.NPFTDT6 == 1,'TauPath'] = 1
aut.loc[aut.NPFTDT7 == 1,'TauPath'] = 1
aut.loc[aut.NPFTDT9 == 1,'TauPath'] = 1
aut.loc[aut.NPFRONT == 1,'TauPath'] = 1
aut.loc[aut.NPTAU == 1,'TauPath'] = 1
aut.loc[aut.ADPath == 1, 'TauPath'] = 0
aut.loc[aut.NPCFTLD == 1, 'TauPath'] = 0

# Code Lewy body disease based on NPLBOD variable. Do not include amygdala-
# predominant, brainstem-predominant, or olfactory-only cases.
# See Toledo et al. (2016, Acta Neuropathol) and Irwin et al. (2018, Nat Rev
# Neuro).
aut = aut.assign(LBPath = [0 for i in range(aut.shape[0])])
aut.loc[aut.NPLBOD.isin((2,3)),'LBPath'] = 1
aut.loc[aut.NPPLEWY == 1,'LBPath'] = 1
aut.loc[aut.NPPLEWY == 2,'LBPath'] = 0
aut.loc[aut.NPCLEWY == 1,'LBPath'] = 0
aut.loc[aut.ADPath == 1 & (aut.NPPLEWY != 1), 'LBPath'] = 0
aut.loc[aut.TauPath == 1 & (aut.NPPLEWY != 1),'LBPath'] = 0

# Code TDP-43 pathology based on NPFTDTDP and NPALSMND, excluding FUS and SOD1
# cases.
aut = aut.assign(TDPPath = [0 for i in range(aut.shape[0])])
aut.loc[aut.NPFTD == 1,'TDPPath'] = 1
aut.loc[aut.NPFTDTDP == 1,'TDPPath'] = 1
aut.loc[aut.NPALSMND == 1,'TDPPath'] = 1
aut.loc[aut.ADPath == 1, 'TDPPath'] = 0
aut.loc[aut.LBPath == 1, 'TDPPath'] = 0
aut.loc[aut.TauPath == 1, 'TDPPath'] = 0

# Code vascular disease based on relevant derived variables:
aut = aut.assign(VPath = [0 for i in range(aut.shape[0])])
aut.loc[aut.NPINF == 1,'VPath'] = 1
aut.loc[aut.NACCMICR == 1,'VPath'] = 1
aut.loc[aut.NACCHEM == 1,'VPath'] = 1
aut.loc[aut.NPPATH == 1,'VPath'] = 1
aut.loc[aut.NPPVASC == 1,'VPath'] = 1
aut.loc[aut.NPPVASC == 2,'VPath'] = 0
aut.loc[aut.NPCVASC == 1,'VPath'] = 0
aut.loc[aut.ADPath == 1 & (aut.NPPVASC != 1), 'VPath'] = 0
aut.loc[aut.LBPath == 1 & (aut.NPPVASC != 1), 'VPath'] = 0
aut.loc[aut.NPPFTLD == 1 & (aut.NPPVASC != 1),'VPath'] = 0
aut.loc[aut.TDPPath == 1 & (aut.NPPVASC != 1), 'VPath'] = 0
aut.loc[aut.TauPath == 1 & (aut.NPPVASC != 1), 'VPath'] = 0

aut = aut.assign(Class = aut.ADPath)
aut.loc[aut.TauPath == 1,'Class'] = 2
aut.loc[aut.TDPPath == 1,'Class'] = 3
aut.loc[aut.LBPath == 1,'Class'] = 4
aut.loc[aut.VPath == 1,'Class'] = 5
aut = aut.loc[aut.Class != 0]
aut.index = list(range(aut.shape[0]))

## Predictor variable preparation: one-hot-encoding, date/age/interval operations,
# consolidating redundant variables, consolidating free-text variables.
aut = aut.assign(DOB = aut.BIRTHYR)
aut = aut.assign(DOD = aut.NACCYOD)
aut = aut.assign(VISITDATE = aut.VISITYR)
for i in range(aut.shape[0]):
    aut.loc[i,'DOB'] = datetime.datetime.strptime('-'.join([str(aut.BIRTHYR.loc[i]),str(aut.BIRTHMO.loc[i]),'01']),'%Y-%m-%d')
    aut.loc[i,'DOD'] = datetime.datetime.strptime('-'.join([str(aut.NACCYOD.loc[i]),str(aut.NACCMOD.loc[i]),'01']),'%Y-%m-%d')
    aut.loc[i,'VISITDATE'] = datetime.datetime.strptime('-'.join([str(aut.VISITYR.loc[i]),str(aut.VISITMO.loc[i]),str(aut.VISITDAY.loc[i])]),'%Y-%m-%d')

# Some time/interval variables
aut = aut.assign(SinceQUITSMOK = aut.NACCAGE - aut.QUITSMOK) # Years since quitting smoking
aut = aut.assign(AgeStroke = aut.NACCSTYR - aut.BIRTHYR)
aut = aut.assign(AgeTIA = aut.NACCTIYR - aut.BIRTHYR)
aut = aut.assign(AgePD = aut.PDYR - aut.BIRTHYR)
aut = aut.assign(AgePDOTHR = aut.PDOTHRYR - aut.BIRTHYR)
aut = aut.assign(AgeTBI = aut.TBIYEAR - aut.BIRTHYR)
aut = aut.assign(Duration = aut.NACCAGE - aut.DECAGE)

# Hispanic origin
aut.HISPORX = aut.HISPORX.str.lower()
aut.loc[aut.HISPORX == 'spanish','HISPORX'] = 'spain'

# Race. RACESECX and RACETERX have too few values to be useful.
aut.RACEX = aut.RACEX.str.lower().str.replace(' ','').str.replace('-','')
aut.loc[aut.RACEX.isin(['hispanic','puerto rican']),'RACEX'] = 'latino'
aut.loc[aut.RACEX.isin(['guam - chamorro']),'RACEX'] = 'chamorro'
aut.loc[aut.RACEX.isin(['multi racial']),'RACEX'] = 'multiracial'

# Other language. But actually, let's just drop this and code as English/non-English.
#aut.PRIMLANX = aut.PRIMLANX.str.lower().str.replace(' ','').str.replace('-','')

# Drug list. First get a list of all the unique drug names, then code as dummy variables.
# Update as of 04/01/2020: drugs alone are going to be a huge amount of work.
# For now, just rely on the NACC derived variables for diabetes meds, cardiac drugs, etc.
drugcols = ['DRUG' + str(i) for i in range(1,41)]
drugs = aut[drugcols].stack()
# Several varieties of insulin--important to distinguish?
# drop "*not-codable"
# drop "diphtheria/hepb/pertussis,acel/polio/tetanus"
drugs = drugs.unique()
drugs = [eachdrug.lower() for eachdrug in drugs.tolist()]
drugs = pd.Series(drugs)
drug_corrections = [("multivitamin with minerals","multivitamin"),
    ("multivitamin, prenatal","multivitamin"),
    ("omega 3-6-9","omega369"),
    ("omega-3","omega3"),
    ("vitamin-d","vitamin d"),
    ("acetyl-l-carnitine","acetyl l carnitine"),
    ("levodopa","levadopa"),
    ("pro-stat","prostat"),
    ("alpha-d-galactosidase","alpha d galactosidase"),
    ("indium pentetate in-111","indium pentetate in111"),
    ("fludeoxyglucose f-18","fludeoxyglucose f18"),
    ("calcium with vitamins d and k", "calcium-vitamin d-vitamin k"),
    ("aloe vera topical", "aloe vera"),
    ("ammonium lactate topical", "ammonium lactate")]
for i in range(len(drug_corrections)):
    oldval = drug_corrections[i][0]
    newval = drug_corrections[i][1]
    drugs = drugs.str.replace(pat = oldval, repl = newval)

drugs = drugs.loc[drugs != "*not codable*"]
drugs = drugs.loc[drugs != "diphtheria/hepb/pertussis,acel/polio/tetanus"]
drugs = np.unique([ss for eachdrug in drugs for ss in eachdrug.split('-')])
drugs = np.unique([ss for eachdrug in drugs for ss in eachdrug.split('/')])
drugs.sort()

## Combining redundant variables. Often this reflects a change in form or
# variable name between UDS version 2 & 3.
aut.loc[(aut.CVPACE == -4) & (aut.CVPACDEF == 0),'CVPACE'] = 0
aut.loc[(aut.CVPACE == -4) & (aut.CVPACDEF == 1),'CVPACE'] = 1
xvar.loc[xvar.Variable == 'CVPACDEF','Keep'] = False

# Combine TBIBRIEF and TRAUMBRF.
aut.loc[(aut.TBIBRIEF == -4) & (aut.TRAUMBRF.isin([0])),'TBIBRIEF'] = 0
aut.loc[(aut.TBIBRIEF == -4) & (aut.TRAUMBRF.isin([1,2])),'TBIBRIEF'] = 1
xvar.loc[xvar.Variable == 'TRAUMBRF','Keep'] = False

# More data cleaning
aut.ABRUPT = aut.ABRUPT.replace(to_replace = 2, value = 1)
aut.FOCLSYM = aut.FOCLSYM.replace(to_replace = 2, value = 1)
aut.FOCLSIGN = aut.FOCLSIGN.replace(to_replace = 2, value = 1)

# Convert language to a binary variable (English/non-English)
aut = aut.assign(English = 0)
aut.loc[aut.PRIMLANG == 1,'English'] = 1
xvar.loc[xvar.Variable == 'PRIMLANG','Keep'] = False

# Some dummy coding
vv = xvar.Variable.loc[(xvar.Keep) & (xvar.Comments == "Dummy coding for (95,96,97,98)")]
for v in vv:
    aut[v + '_couldnt'] = 0
    aut.loc[aut[v].isin([95,96,97,98]),v + '_couldnt'] = 1
vv = xvar.Variable.loc[xvar.Comments == "Dummy coding for (995,996,997,998)"]
for v in vv:
    aut[v + '_couldnt'] = 0
    aut.loc[aut[v].isin([995,996,997,998]),v + '_couldnt'] = 1

# Drop all columns where xvar.Keep == False.
aut2 = aut
xvar.loc[xvar.Variable == 'NACCID','Keep'] = True
xvar.loc[xvar.Variable == 'NACCID','Type'] = "ID"
xvar.loc[xvar.Variable == 'VISITDATE','Keep'] = True
xvar.loc[xvar.Variable == 'VISITDATE','Type'] = "ID"
aut = aut.drop(columns = xvar.Variable[~xvar.Keep])

# Fill with NA values
xvar = xvar.loc[xvar.Keep]
xvar.index = range(xvar.shape[0])
for i in range(xvar.shape[0]):
    if not xvar.NaNValues.isna()[i]:
        v = xvar.Variable[i]
        badval = eval(xvar.NaNValues[i])
        #print(v,badval)
        if isinstance(badval,int):
            badval = [badval]
        aut[v].mask(aut[v].isin(badval),inplace = True)

# Get rid of variables with very few meaningful observations.
valcounts = aut.describe().iloc[0]
aut = aut.drop(columns = valcounts.loc[valcounts < 100].index)
#aut = aut[valcounts.loc[valcounts >= 100].index]

# Find correlated variables and drop.
ac = aut.corr()
acs = ac.unstack(level = 0)
acs = acs.loc[abs(acs)>0.8]
acsind = list(acs.index)
diagnames = [ind for ind in acsind if ind[0] == ind[1]]
acs = acs.drop(labels=diagnames)
acs = pd.DataFrame(acs)
acs.columns = ['r']
acs['v1'] = acs.index
acs[['v1','v2']] = pd.DataFrame(acs['v1'].tolist(),index = acs.index)

y = aut.Class
X = aut.drop(columns = npvar.Variable.loc[npvar.Variable.isin(aut.columns)])
X = X.drop(columns = ['Class','ADPath','TauPath','TDPPath','LBPath','VPath'])
xd = X.describe().iloc[0]

# Impute numeric variables with the mean.
from sklearn.impute import SimpleImputer
numvar = X.columns.intersection(xvar.Variable.loc[xvar.Type == "Numeric"])
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X[numvar])
Xnumimp = imp_mean.transform(X[numvar])
Xnumimp = pd.DataFrame(Xnumimp)
Xnumimp.columns = X[numvar].columns

# Impute ordinal variables with the median.
ordvar = X.columns.intersection(xvar.Variable.loc[xvar.Type == "Ordinal"])
imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
imp_med.fit(X[ordvar])
Xordimp = imp_med.transform(X[ordvar])
Xordimp = pd.DataFrame(Xordimp)
Xordimp.columns = X[ordvar].columns

# Impute boolean variables with zero.
boolvar = X.columns.intersection(xvar.Variable.loc[xvar.Type == "Boolean"])
boolenc = SimpleImputer(missing_values = np.nan, strategy = 'constant',
    fill_value = 0)
boolenc.fit(X[boolvar])
Xbool = boolenc.transform(X[boolvar])
Xbool = pd.DataFrame(Xbool)
Xbool.columns = X[boolvar].columns

# One-hot encoding for nominal (not boolean, ordinal, or numeric) variables.
from sklearn.preprocessing import OneHotEncoder
nomvar = X.columns.intersection(xvar.Variable.loc[xvar.Type == "Nominal"])
enc = OneHotEncoder(handle_unknown='ignore',sparse = False)
Xfull = X[nomvar].fillna(value = 0)
enc.fit(Xfull)
Xohe = enc.transform(Xfull)
Xohe = pd.DataFrame(Xohe)
Xohe.columns = enc.get_feature_names(Xfull.columns)

# Put it all together
X = X.drop(columns = boolvar)
X = X.drop(columns = numvar)
X = X.drop(columns = ordvar)
X = pd.concat([X,Xbool,Xnumimp,Xordimp,Xohe],axis = 1)
X = X.drop(columns = nomvar)

# Create 80/20 split between data for training and final testing.
# Do data split stratified by pathology class.
from sklearn.model_selection import train_test_split
classy = aut[['Class','SEX','EDUC']]
classy = classy.assign(HighEd = classy.EDUC > 12)
classy = classy.drop(columns = ['EDUC'])
classy = classy.assign(MasterClass = classy.astype(str).apply(lambda x: '_'.join(x),axis = 1))
uclass = np.unique(classy.MasterClass)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=666, stratify=classy.MasterClass)

# Create a further split within the training dataset for CV and for validation.
classy2 = classy.iloc[X_train.index]
X_cv, X_val, y_cv, y_val = train_test_split( X_train, y_train, test_size=0.25, random_state=666, stratify=classy2.MasterClass)

X_cv.index = range(X_cv.shape[0])
y_cv.index = range(y_cv.shape[0])
X_val.index = range(X_val.shape[0])
y_val.index = range(y_val.shape[0])
X_test.index = range(X_test.shape[0])
y_test.index = range(y_test.shape[0])

#import pickle
#PIK = "nacc_train.pkl"
#data = [X_cv,y_cv,X_val,y_val]
#with open(PIK, "wb") as f:
#    pickle.dump(data, f)
#with open(PIK, "rb") as f:
#    pickle_list = pickle.load(f)

# Now load in classifier & classified data to do error analyses.
import pickle
pik = "weovr_classifier_og_data.pickle"
with open(pik, "rb") as f:
    pickle_list = pickle.load(f)

# Here are the contents of the pickle:
#data = [weovr_clf, X_train, X_test, y_train, y_test, OG_X, OG_y, OG_weovr_pred]
wovr = pickle_list[0]
X_aug_train = pickle_list[1]
X_aug_val = pickle_list[2]
y_aug_train = pickle_list[3]
y_aug_val = pickle_list[4]
pikX = pd.DataFrame(pickle_list[5])
feat = pd.read_csv("selected_features.csv")
feat = list(feat.columns)
pikX.columns = feat
piky = pd.DataFrame(pickle_list[6])

wovr_pred = pd.Series(pickle_list[7])

#tmptrain = pd.read_csv("X_cv.csv")
#tmptest = pd.read_csv("X_val.csv")
#tmp = pd.concat([tmptrain,tmptest], axis = 0)
OG_X = pd.concat([X_cv, X_val], axis = 0)
OG_X['WOVR'] = wovr_pred
OG_y = pd.DataFrame(pd.concat([y_cv, y_val], axis = 0))
OG_y += -1
OG_y.columns = ["Class"]
OG_y.index = OG_X.index
#Xy = pd.concat([OG_X, OG_y], axis = 1)
addcol = [*['NACCID','VISITDATE','Class','ADPath','TauPath','TDPPath','LBPath','VPath'], *npvar.Variable.to_list()]
Xy = OG_X.merge(right = aut[addcol], how='inner', on=['NACCID','VISITDATE'],
    indicator='Merge', validate="1:1")
Xy.Class = Xy.Class - 1
#Xy['WOVR'] = wovr_pred

from sklearn.metrics import confusion_matrix
confusion_matrix(Xy.Class, Xy.WOVR, normalize=None)

# Code some additional neuropath measures.
Xy['Braak03'] = np.ceil(Xy.NACCBRAA/2)
Xy.loc[Xy.Braak03 > 3,'Braak03'] = np.nan
thal = [0, 1, 2, 3, 4, 5,-4, 8, 9]
ascore = [0, 1, 1, 2, 3, 3, np.nan, np.nan, np.nan]
adict = dict(zip(thal,ascore))
Xy['Ascore'] = [adict[a] for a in Xy['NPTHAL']]
Xy['Bscore'] = np.ceil(Xy.NACCBRAA/2)
Xy['Cscore'] = Xy.NACCNEUR
Xy.loc[Xy['Cscore'].isin([8,9]), 'Cscore'] = np.nan
Xy['ABC'] = 0
Xy.loc[(Xy['Ascore'] == 1) & (Xy['Cscore'] < 2),'ABC'] = 1
Xy.loc[(Xy['Ascore'] > 0) & (Xy['Bscore'] < 2),'ABC'] = 1
Xy.loc[(Xy['Ascore'] == 1) & (Xy['Bscore'] > 1) & (Xy['Cscore'] > 1) ,'ABC'] = 2
Xy.loc[(Xy['Ascore'] > 1) & (Xy['Bscore'] > 1),'ABC'] = 2
Xy.loc[(Xy['Ascore'] == 3) & (Xy['Bscore'] == 3) & (Xy['Cscore'] > 1) ,'ABC'] = 3

# AD false alarms: people with primary non-AD pathology who were called AD.
print("Distribution of ABC scores for primary non-AD cases who were classified as AD:")
adfa = Xy.loc[(Xy.WOVR == 0) & (Xy.Class != 0),:]
adfatab = pd.crosstab(adfa['Class'],adfa['ABC'])
adfatab.index = ['Tau', 'TDP', 'LB', 'Vasc']
adfatab.to_latex('adfatab.tex')

# Non-AD false alarms: people with primary AD pathology who were called non-AD.
print("Distribution of ABC scores for primary AD cases who were classified as non-AD:")
nadfa = Xy.loc[(Xy.WOVR != 0) & (Xy.Class == 0),:]
pd.crosstab(nadfa['Class'],nadfa['ABC'])

nadfa.loc[nadfa.NPFTDTAU == 1,'TauPath'] = 1
nadfa.loc[nadfa.NACCPICK == 1,'TauPath'] = 1
nadfa.loc[nadfa.NACCCBD == 1,'TauPath'] = 1
nadfa.loc[nadfa.NACCPROG == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPFTDT2 == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPFTDT5 == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPFTDT6 == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPFTDT7 == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPFTDT9 == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPFRONT == 1,'TauPath'] = 1
nadfa.loc[nadfa.NPTAU == 1,'TauPath'] = 1

# Code Lewy body disease based on NPLBOD variable. Do not include amygdala-
# predominant, brainstem-predominant, or olfactory-only cases.
# See Toledo et al. (2016, Acta Neuropathol) and Irwin et al. (2018, Nat Rev
# Neuro).
nadfa.loc[nadfa.NPLBOD.isin((2,3)),'LBPath'] = 1
nadfa.loc[nadfa.NPPLEWY == 1,'LBPath'] = 1
nadfa.loc[nadfa.NPPLEWY == 2,'LBPath'] = 0

# Code TDP-43 pathology based on NPFTDTDP and NPALSMND, excluding FUS and SOD1
# cases.
nadfa.loc[nadfa.NPFTD == 1,'TDPPath'] = 1
nadfa.loc[nadfa.NPFTDTDP == 1,'TDPPath'] = 1
nadfa.loc[nadfa.NPALSMND == 1,'TDPPath'] = 1

# Code vascular disease based on relevant derived variables:
nadfa.loc[nadfa.NPINF == 1,'VPath'] = 1
nadfa.loc[nadfa.NACCMICR == 1,'VPath'] = 1
nadfa.loc[nadfa.NACCHEM == 1,'VPath'] = 1
nadfa.loc[nadfa.NPPATH == 1,'VPath'] = 1
nadfa.loc[nadfa.NPPVASC == 1,'VPath'] = 1

nadfatab = pd.DataFrame(np.stack([ nadfa.TauPath.value_counts(),
    nadfa.TDPPath.value_counts(),
    nadfa.LBPath.value_counts(),
    nadfa.VPath.value_counts() ]))
nadfatab.index = ['Tau','TDP','LB','Vasc']
nadfatab.columns = ['No','Yes']
nadfatab.to_latex('nadfatab.tex')

# Non-AD false alarms: people with primary AD pathology who were called non-AD.
print("Presence of vascular pathology in cases misclassified as primarily vascular:")
vfa = Xy.loc[(Xy.WOVR == 4) & (Xy.Class != 4),:]

vfa['NPINF'] = vfa['NPINF'].replace(to_replace = [-4,8,9], value = np.nan)
vfa['NACCMICR'] = vfa['NACCMICR'].replace(to_replace = [-4,8,9], value = np.nan)
vfa['NACCHEM'] = vfa['NACCHEM'].replace(to_replace = [-4,8,9], value = np.nan)
vfa['NACCMICR'] = vfa['NACCMICR'].replace(to_replace = [-4,8,9], value = np.nan)
vfa['NPPATH'] = vfa['NPPATH'].replace(to_replace = [-4,8,9], value = np.nan)
vfa['NPPVASC'] = vfa['NPPVASC'].replace(to_replace = [2], value = 0)
vfa['NPPVASC'] = vfa['NPPVASC'].replace(to_replace = [-4,8,9], value = np.nan)

vfa.loc[vfa.NPINF == 1,'VPath'] = 1
vfa.loc[vfa.NACCMICR == 1,'VPath'] = 1
vfa.loc[vfa.NACCHEM == 1,'VPath'] = 1
vfa.loc[vfa.NPPATH == 1,'VPath'] = 1
vfa.loc[vfa.NPPVASC == 1,'VPath'] = 1

vfatab = pd.DataFrame(np.stack([ vfa.NPPVASC.value_counts(),
    vfa.NPINF.value_counts(),
    vfa.NACCMICR.value_counts(),
    vfa.NACCHEM.value_counts(),
    vfa.NPPATH.value_counts() ]))
vfatab.index = ['Primary vascular','Old infarcts', 'Microinfarcts','Hemorrhages','Other']
vfatab.columns = ['No','Yes']
vfatab.to_latex('vfatab.tex')
