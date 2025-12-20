import os
import sys
import requests
from urllib.parse import urljoin

# put repo root on path for consistency
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

BASE = os.getenv('CEDA_BASE_URL', 'https://api.ceda.ashoka.edu.in/v1')
API_KEY = os.getenv('CEDA_API_KEY', '')
KEY_MODE = os.getenv('CEDA_KEY_MODE', 'header').lower()

candidates = [
    '',
    'commodities', 'commodity', 'prices', 'price', 'markets', 'market', 'data',
    'api/commodities', 'api/markets', 'api/prices', 'api/pricepoints',
    'search/commodities', 'commodities/search', 'commodity/prices', 'market/prices'
]

# Build a short list of likely base URLs to probe (deduplicated)
bases = [BASE]
variants = [
    BASE.replace('/v1', ''),
    'https://ceda.ashoka.edu.in',
    'https://ceda.ashoka.edu.in/api',
    'https://ceda.ashoka.edu.in/api/v1',
    'https://ceda.ashoka.edu.in/api/v0',
    'https://api.ceda.ashoka.edu.in/api'
]
for v in variants:
    if v and v not in bases:
        bases.append(v)

headers = {'Accept': 'application/json'}
if API_KEY and KEY_MODE == 'header':
    headers['Authorization'] = f'Bearer {API_KEY}'

print('Probing CEDA base URLs:', bases)
print('Using API key present:', bool(API_KEY), 'KEY_MODE:', KEY_MODE)

results = []
for b in bases:
    for cand in candidates:
        full = b.rstrip('/') + ('/' + cand if cand else '')
        params = {}
        if API_KEY and KEY_MODE == 'query':
            params['api_key'] = API_KEY
        try:
            r = requests.get(full, headers=headers, params=params, timeout=8)
            text_preview = ''
            try:
                text_preview = r.text[:400]
            except Exception:
                text_preview = '<no body>'
            print(f'URL: {full}  -> {r.status_code}')
            print('  body preview:', text_preview.replace('\n', ' ')[:400])
            results.append((full, r.status_code, text_preview))
        except Exception as e:
            print(f'URL: {full}  -> Exception: {e}')
            results.append((full, 'EX', str(e)))

# Summary
print('\nSummary (first 10 results):')
for item in results[:10]:
    print(item[0], item[1])

# Exit code 0 always; user will inspect output
