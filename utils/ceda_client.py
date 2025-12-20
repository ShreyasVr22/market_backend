
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CEDAClient:
    # Base URL can be overridden via env for testing
    BASE_URL = os.getenv("CEDA_BASE_URL", "https://api.ceda.ashoka.edu.in/v1")
    # KEY_MODE: 'header' (default) or 'query'
    KEY_MODE = os.getenv("CEDA_KEY_MODE", "header").lower()

    def __init__(self):
        # Read API key from environment. Do NOT hardcode secrets in source.
        self.api_key = os.getenv("CEDA_API_KEY", "")

    def _build_headers(self):
        headers = {"Accept": "application/json"}
        if self.api_key and self.KEY_MODE == "header":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _attach_key_to_params(self, params):
        if self.api_key and self.KEY_MODE == "query":
            p = params.copy()
            p["api_key"] = self.api_key
            return p
        return params

    def get_current_prices(self, commodity, market=None, limit=100, granularity="daily", **extra):
        """Fetch current prices from CEDA API

        Parameters:
        - commodity: name or code
        - market: market name or code
        - limit: number of records to return
        - granularity: 'daily' (provider-dependent)
        - extra: passthrough provider-specific params (state, district, variety, start_date, end_date)
        """
        try:
            endpoint = f"{self.BASE_URL}/commodities"
            params = {"commodity": commodity, "limit": limit, "granularity": granularity}

            if market:
                params["market"] = market

            # include extras
            params.update(extra)

            # attach key (query) if needed
            params = self._attach_key_to_params(params)
            headers = self._build_headers()

            response = requests.get(endpoint, params=params, headers=headers, timeout=12)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ CEDA API: Fetched {len(data) if hasattr(data, '__len__') else 'result'} records")
                return data
            else:
                body = ""
                try:
                    body = response.text
                except Exception:
                    pass
                print(f"✗ CEDA API Error: {response.status_code} {body}")
                return None

        except Exception as e:
            print(f"✗ CEDA API Exception: {e}")
            return None

    def get_all_markets(self):
        """Fetch all available markets"""
        try:
            endpoint = f"{self.BASE_URL}/markets"
            headers = self._build_headers()
            response = requests.get(endpoint, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            print(f"✗ Error fetching markets: {e}")
            return None

    def get_all_commodities(self):
        """Fetch all available commodities"""
        try:
            endpoint = f"{self.BASE_URL}/commodities"
            headers = self._build_headers()
            response = requests.get(endpoint, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            print(f"✗ Error fetching commodities: {e}")
            return None
