import os
import json
from models.data_loader import DataLoader

def main():
    csv_path = os.environ.get('CSV_PATH')
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit('CSV_PATH not set or file missing')

    loader = DataLoader(csv_path)
    loader.load_csv()

    market = os.environ.get('INSPECT_MARKET', 'Bangalore')

    df = loader.filter_by_market_crop(market, None)
    out = {'market': market, 'rows': 0, 'market_names': [], 'variety_sample': [], 'group_sample': []}
    if df is None:
        print('NO_DATA')
    else:
        out['rows'] = int(len(df))
        out['market_names'] = list(map(str, df['Market Name'].astype(str).unique()[:50]))
        if 'Variety' in df.columns:
            out['variety_sample'] = list(map(str, sorted(set(df['Variety'].astype(str).unique()))[:100]))
        if 'Group' in df.columns:
            out['group_sample'] = list(map(str, sorted(set(df['Group'].astype(str).unique()))[:100]))

    with open('inspect_market_Bangalore.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('WROTE inspect_market_Bangalore.json')

if __name__ == '__main__':
    main()
