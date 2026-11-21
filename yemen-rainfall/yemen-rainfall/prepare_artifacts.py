"""Creates history.csv (+ pcode_map.json, scalers.json if you do not already have them)
from the original rainfall CSV. Usage:  python prepare_artifacts.py [path_to_csv]
Default CSV name: Yemen_Rainfall_1982_2025.csv (must contain date, PCODE, rfh, r1h, r3h)."""
import json, os, sys, numpy as np, pandas as pd
FEATURES = ['rfh','r1h','r3h','rfh_lag_1','rfh_lag_2','rfh_lag_3','rfh_lag_6','rfh_lag_12',
            'month_sin','month_cos','dekad_sin','dekad_cos','month','dekad']
path = sys.argv[1] if len(sys.argv) > 1 else 'Yemen_Rainfall_1982_2025.csv'
if not os.path.exists(path):
    sys.exit(f'CSV not found: {path}\nPut your rainfall CSV in this folder or run: python prepare_artifacts.py "C:\\\\path\\\\file.csv"')
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]
need = ['date', 'PCODE', 'rfh', 'r1h', 'r3h']
if [c for c in need if c not in df.columns]: sys.exit(f'Missing columns {[c for c in need if c not in df.columns]}. Found: {list(df.columns)}')
d = pd.to_datetime(df['date'])
df['date'] = d.dt.strftime('%Y-%m-%d'); df['year'] = d.dt.year; df['month'] = d.dt.month
df['dekad'] = d.dt.day.map(lambda x: 1 if x <= 10 else 11 if x <= 20 else 21)       # 1/11/21 as in training
df = df.sort_values(['PCODE', 'year', 'month', 'dekad']).reset_index(drop=True)
df[['date', 'PCODE', 'rfh', 'r1h', 'r3h']].to_csv('history.csv', index=False)        # raw, un-logged
print('wrote history.csv', len(df), 'rows,', df['PCODE'].nunique(), 'PCODEs')
if not os.path.exists('pcode_map.json'):
    cats = df['PCODE'].astype('category')
    json.dump({str(c): i for i, c in enumerate(cats.cat.categories)}, open('pcode_map.json', 'w')); print('wrote pcode_map.json')
if os.path.exists('x_scaler.pkl') or os.path.exists('scalers.json'):
    sys.exit('Scalers already present, kept as they are.')
for l in [1,2,3,6,12]: df[f'rfh_lag_{l}'] = df.groupby('PCODE')['rfh'].shift(l)
df = df.dropna()
df['month_sin']=np.sin(2*np.pi*df['month']/12); df['month_cos']=np.cos(2*np.pi*df['month']/12)
df['dekad_sin']=np.sin(2*np.pi*df['dekad']/3);  df['dekad_cos']=np.cos(2*np.pi*df['dekad']/3)
for c in ['rfh','r1h','r3h']: df[c]=np.log1p(df[c])
tr = df[df['date'] <= '2015-12-31']                                                   # train split used for fitting
json.dump({'features':FEATURES,'x_min':tr[FEATURES].min().tolist(),'x_max':tr[FEATURES].max().tolist(),
           'y_min':float(tr['rfh'].min()),'y_max':float(tr['rfh'].max())}, open('scalers.json','w')); print('wrote scalers.json')
