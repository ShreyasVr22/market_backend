import time
import requests
import sys

BASE = "http://127.0.0.1:8001/api/market"

def wait_for_health(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:8001/health", timeout=3)
            if r.status_code == 200:
                print("root /health OK:", r.json())
                return True
        except Exception:
            pass
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print("api/market/health OK:", r.json())
                return True
        except Exception:
            pass
        time.sleep(1)
    print("Timeout waiting for server health")
    return False

def main():
    if not wait_for_health(30):
        sys.exit(2)

    # Get districts
    r = requests.get(f"{BASE}/districts", timeout=5)
    print("/districts ->", r.status_code, r.text[:1000])
    data = r.json()
    districts = data.get('districts', [])
    if not districts:
        print("No districts available")
        return

    district = districts[0]
    r = requests.get(f"{BASE}/district/{district}", timeout=5)
    print(f"/district/{district} ->", r.status_code, r.text[:1000])
    ddata = r.json()
    markets = ddata.get('markets', [])
    crops = ddata.get('crops', [])
    if not markets or not crops:
        print("No markets or crops for district", district)
        return

    market = markets[0]
    crop = crops[0]
    print(f"Using market={market!r} crop={crop!r} for price check")

    # Query prices endpoint
    mr = requests.get(f"{BASE}/prices/{market}/{crop}", timeout=10)
    print(f"/prices/{market}/{crop} ->", mr.status_code)
    try:
        print(mr.json())
    except Exception:
        print(mr.text)

if __name__ == '__main__':
    main()
