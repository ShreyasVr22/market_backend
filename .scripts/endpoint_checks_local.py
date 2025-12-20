import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from fastapi.testclient import TestClient
from main import app

def run_checks():
    with TestClient(app) as client:
        # root health
        r = client.get("/health")
        print("root /health ->", r.status_code, r.json())

        # api/market health
        r = client.get("/api/market/health")
        print("/api/market/health ->", r.status_code, r.json())

        # districts
        r = client.get("/api/market/districts")
        print("/api/market/districts ->", r.status_code)
        data = r.json()
        print("districts count:", len(data.get('districts', [])))

        districts = data.get('districts', [])
        if not districts:
            print("No districts available")
            return

        district = districts[0]
        r = client.get(f"/api/market/district/{district}")
        print(f"/api/market/district/{district} ->", r.status_code)
        ddata = r.json()
        markets = ddata.get('markets', [])
        crops = ddata.get('crops', [])
        print("markets", len(markets), "crops", len(crops))

        if not markets or not crops:
            print("No markets or crops for district", district)
            return

        market = markets[0]
        crop = crops[0]
        print(f"Using market={market!r} crop={crop!r} for price check")

        mr = client.get(f"/api/market/prices/{market}/{crop}")
        print(f"/api/market/prices/{market}/{crop} ->", mr.status_code)
        try:
            print(mr.json())
        except Exception:
            print(mr.text)

if __name__ == '__main__':
    run_checks()
