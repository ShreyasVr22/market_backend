import json
import os
import sys

# ensure project root is on sys.path
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from routes import market

# Ensure CSV and DISTRICTS_DATA are built
market.startup_load()
results = {}
results['health'] = market.health()

d = market.get_districts()
results['districts'] = d

first = d.get('districts', [None])[0]
if first:
    try:
        dm = market.get_district_markets(first)
        results['sample_district'] = dm
    except Exception as e:
        results['sample_district_error'] = str(e)
else:
    results['sample_district'] = None

out_path = os.path.join(repo_root, 'scripts', 'check_endpoints_output.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f'WROTE_OUTPUT: {out_path}')
