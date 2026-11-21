"""iTransformer backend. Standard library + (torch, numpy, pandas). No Flask needed.
Run:  python server.py   then open http://localhost:8000
It starts even when files are missing (demo mode) and loads the model as soon as they appear.
Needed in this folder for real predictions:
  best_itransformer__2_.pth (or any *.pth), scalers.json, history.csv, pcode_map.json  (from prepare_artifacts.py)
  model_config.json -> {"n_heads": N}   (Optuna best_params["n_heads"]; not stored in the weights)"""
import glob, json, math, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
SEQ, NF = 24, 14
FEATURES = ['rfh','r1h','r3h','rfh_lag_1','rfh_lag_2','rfh_lag_3','rfh_lag_6','rfh_lag_12','month_sin','month_cos','dekad_sin','dekad_cos','month','dekad']
CONTRACT = '1.0'
ST = {'ready': False, 'missing': [], 'error': None}

def build():
    global SEQ
    import numpy as np, pandas as pd, torch, torch.nn as nn
    class iTransformer(nn.Module):   # identical to the training notebook
        def __init__(self, num_features, seq_len, num_pcodes, d_model=128, n_heads=4, num_layers=3, d_ff=256, dropout=0.1, pcode_embed_dim=16):
            super().__init__()
            self.pcode_emb = nn.Embedding(num_pcodes, pcode_embed_dim)
            self.value_embedding = nn.Linear(seq_len, d_model)
            self.feature_embedding = nn.Parameter(torch.randn(1, num_features, d_model))
            self.total_dim = d_model + pcode_embed_dim
            layer = nn.TransformerEncoderLayer(d_model=self.total_dim, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.norm = nn.LayerNorm(self.total_dim)
            self.head = nn.Sequential(nn.Linear(self.total_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
        def forward(self, x, pcode):
            B, T, F = x.shape
            x = self.value_embedding(x.permute(0, 2, 1)) + self.feature_embedding
            x = torch.cat([x, self.pcode_emb(pcode).unsqueeze(1).expand(-1, F, -1)], dim=-1)
            x = self.encoder(self.norm(x)).mean(dim=1)
            return self.head(x).squeeze()
    pth = sorted(glob.glob('*.pth'))
    have_sc = os.path.exists('scalers.json') or (os.path.exists('x_scaler.pkl') and os.path.exists('y_scaler.pkl'))
    miss = [f for f in ('history.csv', 'pcode_map.json', 'model_config.json') if not os.path.exists(f)]
    if not have_sc: miss.append('scalers.json (or x_scaler.pkl + y_scaler.pkl)')
    if not pth: miss.append('best_itransformer.pth')
    if miss: return miss
    cfg = json.load(open('model_config.json'))
    sd = torch.load(pth[0], map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    # Read the architecture from the checkpoint itself, so any trained size loads correctly
    d_model, seq_len = sd['value_embedding.weight'].shape
    NF_ck = sd['feature_embedding'].shape[1]
    n_pc, pc_dim = sd['pcode_emb.weight'].shape
    layers = 1 + max(int(k.split('.')[2]) for k in sd if k.startswith('encoder.layers.'))
    d_ff = sd['encoder.layers.0.linear1.weight'].shape[0]
    total = d_model + pc_dim; heads = int(cfg['n_heads'])
    if total % heads: raise ValueError(f'n_heads={heads} must divide {total} (d_model {d_model} + pcode embed {pc_dim}).')
    if NF_ck != NF: raise ValueError(f'Checkpoint has {NF_ck} features, app expects {NF}.')
    SEQ = int(seq_len)
    ST['arch'] = dict(d_model=int(d_model), pcode_embed_dim=int(pc_dim), num_layers=layers, d_ff=int(d_ff), n_heads=heads, seq_len=SEQ, num_pcodes=int(n_pc))
    m = iTransformer(NF, SEQ, int(n_pc), d_model=int(d_model), n_heads=heads, num_layers=layers, d_ff=int(d_ff), pcode_embed_dim=int(pc_dim))
    m.load_state_dict(sd); m.eval(); print('Loaded iTransformer:', ST['arch'])
    if not (os.path.exists('x_scaler.pkl') and os.path.exists('y_scaler.pkl')): S = json.load(open('scalers.json'))   # .pkl from training wins if present
    else:
        import joblib; xs, ys = joblib.load('x_scaler.pkl'), joblib.load('y_scaler.pkl')   # MinMaxScaler, range (0,1)
        S = {'features': FEATURES, 'x_min': list(xs.data_min_), 'x_max': list(xs.data_max_), 'y_min': float(ys.data_min_[0]), 'y_max': float(ys.data_max_[0])}
    ST.update(np=np, pd=pd, torch=torch, model=m, S=S, XMIN=np.array(S['x_min']), XMAX=np.array(S['x_max']),
              PMAP=json.load(open('pcode_map.json')), H=pd.read_csv('history.csv', parse_dates=['date']))
    return []

def ensure():
    if ST['ready']: return
    try:
        ST['missing'] = build(); ST['error'] = None; ST['ready'] = not ST['missing']
    except Exception as e:
        ST['error'] = f'{type(e).__name__}: {e}'; ST['ready'] = False

def dekad(d): return 1 if d.day <= 10 else 11 if d.day <= 20 else 21
def next_dekad(d):
    if d.day <= 10: return d.replace(day=11)
    if d.day <= 20: return d.replace(day=21)
    return (d.replace(day=1) + ST['pd'].offsets.MonthBegin(1)).normalize()

def predict(j):
    np, pd, torch, S = ST['np'], ST['pd'], ST['torch'], ST['S']
    if str(j['pcode']) not in ST['PMAP']: raise ValueError('PCODE is not in the training mapping.')
    if min(j['rfh'], j['r1h'], j['r3h']) < 0: raise ValueError('Rainfall must be non-negative.')
    date = pd.Timestamp(j['date']); date = date.replace(day=dekad(date))   # snap to dekad start: 1, 11 or 21
    last = ST['H'][ST['H'].PCODE == int(j['pcode'])].date.max()
    if date > next_dekad(last): raise ValueError(f'History for PCODE {j["pcode"]} ends {last.date()}. The latest date you can use is {next_dekad(last).date()}.')
    g = ST['H'][(ST['H'].PCODE == int(j['pcode'])) & (ST['H'].date < date)].sort_values('date').tail(SEQ + 11)
    g = pd.concat([g, pd.DataFrame([{'date': date, 'PCODE': j['pcode'], 'rfh': j['rfh'], 'r1h': j['r1h'], 'r3h': j['r3h']}])], ignore_index=True)
    if len(g) < SEQ + 12: raise ValueError(f'Need {SEQ+11} earlier dekads of history for this PCODE; found {len(g)-1}.')
    for a, b in zip(g.date[:-1], g.date[1:]):
        if next_dekad(a) != b: raise ValueError(f'Gap in history for PCODE {j["pcode"]} between {a.date()} and {b.date()} (dates used: {date.date()} snapped to a dekad start).')
    for l in (1, 2, 3, 6, 12): g[f'rfh_lag_{l}'] = g['rfh'].shift(l)      # raw values, as in training
    g = g.tail(SEQ).copy()
    g['month'] = g.date.dt.month; g['dekad'] = g.date.map(dekad)
    g['month_sin'] = np.sin(2*np.pi*g.month/12); g['month_cos'] = np.cos(2*np.pi*g.month/12)
    g['dekad_sin'] = np.sin(2*np.pi*g.dekad/3);  g['dekad_cos'] = np.cos(2*np.pi*g.dekad/3)
    for c in ('rfh', 'r1h', 'r3h'): g[c] = np.log1p(g[c])
    X = (g[S['features']].values - ST['XMIN']) / (ST['XMAX'] - ST['XMIN'])
    with torch.no_grad():
        y = float(ST['model'](torch.tensor(X[None], dtype=torch.float32), torch.tensor([int(ST['PMAP'][str(j['pcode'])])])))
    mm = max(0.0, math.expm1(y * (S['y_max'] - S['y_min']) + S['y_min']))
    return {'contract_version': CONTRACT, 'model': 'iTransformer', 'predicted_mm': mm, 'unit': 'mm per dekad', 'target_date': str(next_dekad(date).date()), 'window': {'length': SEQ, 'end': str(date.date())}}

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        self.send_header('Access-Control-Allow-Origin', '*'); self.send_header('Access-Control-Allow-Headers', 'Content-Type'); self.end_headers(); self.wfile.write(b)
    def do_OPTIONS(self): self._send(204, b'')
    def do_GET(self):
        if self.path.startswith('/api/health'):
            ensure(); return self._send(200, {'contract_version': CONTRACT, 'model': 'iTransformer', 'seq_len': SEQ, 'model_loaded': ST['ready'], 'missing': ST['missing'], 'error': ST['error']})
        if self.path in ('/', '/index.html'): return self._send(200, open('index.html', 'rb').read(), 'text/html; charset=utf-8')
        self._send(404, {'error': 'Not found'})
    def do_POST(self):
        if self.path != '/api/predict': return self._send(404, {'error': 'Not found'})
        ensure()
        if not ST['ready']: return self._send(503, {'code': 'MODEL_NOT_LOADED', 'error': 'Model not loaded. Missing: ' + ', '.join(ST['missing'] or [ST['error'] or 'unknown'])})
        try: self._send(200, predict(json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        except (ValueError, KeyError, TypeError) as e: self._send(400, {'code': 'VALIDATION_ERROR', 'error': str(e)})
        except Exception as e: self._send(500, {'code': 'INFERENCE_ERROR', 'error': str(e)})

if __name__ == '__main__':
    ensure()
    print('Model ready.' if ST['ready'] else f"Demo mode. Missing: {ST['missing']} {ST['error'] or ''}")
    print('Open http://localhost:8000'); ThreadingHTTPServer(('0.0.0.0', 8000), H).serve_forever()
