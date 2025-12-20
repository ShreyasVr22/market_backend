import os
import sys

# ensure repo root on sys.path for imports
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.ceda_client import CEDAClient

if __name__ == '__main__':
    client = CEDAClient()
    print('CEDA_API_KEY present in client:', bool(client.api_key))
    # Small smoke call: commodity Tomato, market Bangalore, limit 5
    result = client.get_current_prices(commodity='Tomato', market='Bangalore', limit=5)
    if result is None:
        print('CEDA smoke call returned no data (check API key / network / endpoint).')
    else:
        print('CEDA smoke call returned type:', type(result))
        try:
            if isinstance(result, list):
                print('First item sample:', result[0])
            else:
                print('Result preview:', result)
        except Exception as e:
            print('Could not preview result:', e)
