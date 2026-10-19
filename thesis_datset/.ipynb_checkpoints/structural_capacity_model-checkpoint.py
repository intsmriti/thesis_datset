"""Reproducible scenario-based CBSA regional capacity stress test.
NOT observed diversion, bed occupancy, or geographic accessibility.
"""
import pandas as pd, numpy as np
from pathlib import Path
BASE_DIR = Path.cwd().parent
att=pd.read_csv(P/'final_model_data_verified.csv',dtype={'CCN':str})
att['CCN']=att.CCN.str.zfill(6)
att['attack_year']=pd.to_datetime(att.attack_date).dt.year
parts=[]; audits=[]
for year in range(2016,2022):
    df=pd.read_csv(P/f'CostReport_{year}_Final.csv',low_memory=False,dtype={'Provider CCN':str})
    df['CCN']=df['Provider CCN'].str.strip().str.zfill(6)
    df['year']=year
    df['beds']=pd.to_numeric(df['Number of Beds'],errors='coerce')
    df['cbsa']=pd.to_numeric(df['Medicare CBSA Number'],errors='coerce')
    df['end_date']=pd.to_datetime(df['Fiscal Year End Date'],errors='coerce')
    # Explicit plausibility screen, not a claim that excluded records are necessarily erroneous.
    df['valid_beds']=df.beds.between(1,2000)
    # CMS 999xx codes may denote rural/non-metropolitan designations, not shared local regions.
    df['valid_cbsa']=df.cbsa.between(10000,89999)
    audits.append(dict(year=year,raw_rows=len(df),unique_ccn=df.CCN.nunique(),duplicate_ccn_rows=df.CCN.duplicated().sum(),invalid_beds=(~df.valid_beds).sum(),invalid_cbsa=(~df.valid_cbsa).sum()))
    # Same CCN-year reports: prefer a valid-bed record, then latest fiscal end date.
    df=df.sort_values(['CCN','valid_beds','end_date'],ascending=[True,False,False]).drop_duplicates('CCN')
    parts.append(df[['CCN','year','beds','cbsa','valid_beds','valid_cbsa']])
hosp=pd.concat(parts,ignore_index=True)
joined=att.merge(hosp,left_on=['CCN','attack_year'],right_on=['CCN','year'],how='left',validate='many_to_one')
# Network: all valid alternative hospitals sharing the same numeric metro CBSA in the same year.
network=hosp.loc[hosp.valid_beds & hosp.valid_cbsa].copy()
regional=network.groupby(['year','cbsa']).agg(region_hospitals=('CCN','nunique'),region_beds=('beds','sum')).reset_index()
joined=joined.merge(regional,left_on=['attack_year','cbsa'],right_on=['year','cbsa'],how='left',suffixes=('','_region'),validate='many_to_one')
joined['network_eligible']=joined.valid_beds.fillna(False)&joined.valid_cbsa.fillna(False)&joined.region_hospitals.notna()
joined['alternative_hospitals']=np.where(joined.network_eligible,joined.region_hospitals-1,np.nan)
joined['alternative_beds']=np.where(joined.network_eligible,joined.region_beds-joined.beds,np.nan)
# Drop records with no alternatives: these are still observed, but cannot absorb displaced demand.
scenarios=[]
for occ in [0.70,0.85,0.95]:
 for loss in [0.10,0.25,0.50]:
    z=joined.loc[joined.network_eligible,['CCN','threat_id','attack_year','disruption_severity','beds','alternative_hospitals','alternative_beds']].copy()
    z['assumed_occupancy']=occ;z['assumed_attack_capacity_loss']=loss
    z['displaced_bed_equivalents']=z.beds*loss
    z['hypothetical_spare_alternative_beds']=z.alternative_beds*(1-occ)
    z['unabsorbed_bed_equivalents']=np.maximum(z.displaced_bed_equivalents-z.hypothetical_spare_alternative_beds,0)
    z['unabsorbed_fraction']=z.unabsorbed_bed_equivalents/z.displaced_bed_equivalents
    z['hypothetical_absorption_ratio']=z.hypothetical_spare_alternative_beds/z.displaced_bed_equivalents
    scenarios.append(z)
sc=pd.concat(scenarios,ignore_index=True)
summ=sc.groupby(['assumed_occupancy','assumed_attack_capacity_loss']).agg(n=('CCN','size'),mean_unabsorbed_fraction=('unabsorbed_fraction','mean'),median_unabsorbed_fraction=('unabsorbed_fraction','median'),pct_scenarios_with_shortfall=('unabsorbed_bed_equivalents',lambda s:100*(s>1e-9).mean()),total_displaced_bed_equivalents=('displaced_bed_equivalents','sum'),total_unabsorbed_bed_equivalents=('unabsorbed_bed_equivalents','sum')).reset_index()
# Observed severity association of hypothetical stress metric: exploratory only, no leakage-free predictive claim.
from scipy.stats import spearmanr
assoc=[]
for (occ,loss),g in sc.groupby(['assumed_occupancy','assumed_attack_capacity_loss']):
 rho,p=spearmanr(g.unabsorbed_fraction,g.disruption_severity)
 assoc.append(dict(occupancy=occ,loss=loss,spearman_rho=rho,spearman_p=p,n=len(g)))
pd.DataFrame(audits).to_csv(P/'structural_source_audit.csv',index=False)
joined.to_csv(P/'structural_attack_linkage.csv',index=False)
sc.to_csv(P/'structural_scenarios.csv',index=False)
summ.to_csv(P/'structural_scenario_summary.csv',index=False)
pd.DataFrame(assoc).to_csv(P/'structural_severity_exploratory.csv',index=False)
print('ATTACK_ROWS',len(joined),'MATCHED',joined.beds.notna().sum(),'NETWORK_ELIGIBLE',joined.network_eligible.sum(),'NETWORK_EXCLUDED',(~joined.network_eligible).sum())
print('EXCLUSION DETAILS\n',joined.loc[~joined.network_eligible,['valid_beds','valid_cbsa']].value_counts(dropna=False).to_string())
print('ALTERNATIVE_BEDS',joined.loc[joined.network_eligible,'alternative_beds'].describe().round(1).to_dict())
print('SUMMARY\n',summ.round(3).to_string(index=False))
print('ASSOCIATIONS\n',pd.DataFrame(assoc).round(4).to_string(index=False))
