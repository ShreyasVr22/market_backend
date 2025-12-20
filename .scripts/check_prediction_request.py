import requests
from pprint import pprint

url = 'http://127.0.0.1:8001/api/market/prices/Bangalore/(Whole)'
try:
    r = requests.get(url, timeout=10)
    print('status', r.status_code)
    pprint(r.json())
except Exception as e:
    print('request failed:', e)
