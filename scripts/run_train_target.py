import os
import json
from models.data_loader import DataLoader
from models.trainer import train_and_save_all

def main():
    csv_path = os.environ.get('CSV_PATH')
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit('CSV_PATH not set or file missing')

    target_market = os.environ.get('TARGET_MARKET') or 'Doddaballa Pur'
    target_crop = os.environ.get('TARGET_CROP') or 'Pudi'

    loader = DataLoader(csv_path)
    loader.load_csv()

    districts_data = {
        'Bangalore': {
            'markets': [target_market],
            'crops': [target_crop]
        }
    }

    report = train_and_save_all(loader, districts_data, epochs=3, batch_size=8)
    out_path = os.environ.get('TRAIN_REPORT_PATH', 'train_target_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('WROTE', out_path)

if __name__ == '__main__':
    main()
