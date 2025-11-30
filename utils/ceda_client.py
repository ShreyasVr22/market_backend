
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CEDAClient:
    BASE_URL = "https://api.ceda.ashoka.edu.in/v1"
    
    def __init__(self):
        self.api_key = os.getenv("CEDA_API_KEY", "")
    
    def get_current_prices(self, commodity, market=None):
        """Fetch current prices from CEDA API"""
        try:
            endpoint = f"{self.BASE_URL}/commodities"
            params = {
                "commodity": commodity,
                "limit": 100
            }
            
            if market:
                params["market"] = market
            
            response = requests.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ CEDA API: Fetched {len(data)} records")
                return data
            else:
                print(f"✗ CEDA API Error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"✗ CEDA API Exception: {e}")
            return None
    
    def get_all_markets(self):
        """Fetch all available markets"""
        try:
            endpoint = f"{self.BASE_URL}/markets"
            response = requests.get(endpoint, timeout=10)
            
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
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            return None
        
        except Exception as e:
            print(f"✗ Error fetching commodities: {e}")
            return None
