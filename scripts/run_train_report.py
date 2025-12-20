import json
import traceback
import os
import sys

# Ensure repository root is on sys.path so local packages import correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_loader import DataLoader
from models.trainer import train_and_save_all
import routes.market as rm

def main():
    try:
        import os
        loader = DataLoader(os.getenv('CSV_PATH', r"C:\Users\HP\Downloads\35985678-0d79-46b4-9ed6-6f13308a1d24_201016dc9f0197de809ef278006ca85a.csv"))
        loader.load_csv()
        report = train_and_save_all(loader, rm.DISTRICTS_DATA)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print("Exception during run:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
