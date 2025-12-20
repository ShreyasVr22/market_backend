import os
from pathlib import Path
p = Path(__file__).resolve().parent.parent / 'trained_models'
files = [f.name for f in p.iterdir() if f.is_file()]
h5 = [f for f in files if f.lower().endswith('.h5')]
pkl = [f for f in files if f.lower().endswith('.pkl')]
print('trained_models path:', str(p))
print('h5_count=', len(h5))
print('pkl_count=', len(pkl))
print('sample_h5=', h5[:10])
print('db_exists=', (Path(__file__).resolve().parent.parent / 'market_prices.db').exists())
