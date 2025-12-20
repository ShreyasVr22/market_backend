import os
import sys
import json
import argparse

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_loader import DataLoader
from models.trainer import train_and_save_all


def build_full_mapping(loader):
    df = loader.data
    has_district = 'District' in df.columns
    has_market = 'Market Name' in df.columns
    has_variety = 'Variety' in df.columns
    has_commodity = 'Commodity' in df.columns

    market_to_crops = {}
    district_to_markets = {}

    for _, row in df.iterrows():
        market = str(row['Market Name']).strip() if has_market and row.get('Market Name') else None
        district = str(row['District']).strip() if has_district and row.get('District') else None

        crop = None
        if has_variety and row.get('Variety') and str(row.get('Variety')).strip():
            crop = str(row.get('Variety')).strip()
        elif has_commodity and row.get('Commodity') and str(row.get('Commodity')).strip():
            crop = str(row.get('Commodity')).strip()

        if market and market not in ('nan', 'None'):
            if crop and crop not in ('nan', 'None'):
                market_to_crops.setdefault(market, set()).add(crop)
            if district and district not in ('nan', 'None'):
                district_to_markets.setdefault(district, set()).add(market)
            else:
                # put markets without district under a top-level key
                district_to_markets.setdefault('Unknown', set()).add(market)

    # Build final mapping: district -> {markets: [...], crops: [...]}
    mapping = {}
    for district, markets in district_to_markets.items():
        crops_set = set()
        for m in markets:
            crops_set.update(market_to_crops.get(m, set()))
        mapping[district] = {
            'markets': sorted(list(markets)),
            'crops': sorted(list(crops_set))
        }

    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train models for all markets/crops found in CSV')
    import os
    parser.add_argument('--csv', default=os.getenv('CSV_PATH', r"C:\Users\HP\Downloads\35985678-0d79-46b4-9ed6-6f13308a1d24_201016dc9f0197de809ef278006ca85a.csv"))
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for training (default 10)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', default='trained_models')
    args = parser.parse_args(argv)

    loader = DataLoader(args.csv)
    loader.load_csv()
    if loader.data is None:
        print('No CSV loaded; aborting')
        return

    mapping = build_full_mapping(loader)

    print('Computed mapping summary:')
    print(json.dumps({k: {'markets': len(v['markets']), 'crops': len(v['crops'])} for k, v in mapping.items()}, indent=2))

    # Call train_and_save_all
    report = train_and_save_all(loader, mapping, epochs=args.epochs, batch_size=args.batch_size, out_dir=args.out_dir)

    print('\nTraining report:')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
