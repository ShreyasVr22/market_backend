import os
import sys
import json

# ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.data_loader import DataLoader
import routes.market as rm


def crop_present_in_data(loader, crop, sample_varieties):
    c = crop.lower()
    for v in sample_varieties:
        if c in v.lower():
            return True
    return False


def main():
    import os
    loader = DataLoader(os.getenv('CSV_PATH', r"C:\Users\HP\Downloads\35985678-0d79-46b4-9ed6-6f13308a1d24_201016dc9f0197de809ef278006ca85a.csv"))
    loader.load_csv()

    if loader.data is None:
        print("No CSV loaded; cannot compute mapping.")
        return

    # Gather unique market names and variety/commodity strings
    df = loader.data
    markets_present = set()
    if 'Market Name' in df.columns:
        markets_present = set(df['Market Name'].dropna().astype(str).unique())
    else:
        print('Warning: No Market Name column found in CSV')

    # Use both 'Variety' and 'Commodity' columns if present
    varieties = []
    for col in ['Variety', 'Commodity']:
        if col in df.columns:
            varieties.extend([str(x) for x in df[col].dropna().unique()])

    # Build filtered mapping by keeping only markets/crops that appear
    suggested = {}
    for district, info in rm.DISTRICTS_DATA.items():
        kept_markets = []
        for m in info.get('markets', []):
            ml = m.strip().lower()
            matched = False
            for mp in markets_present:
                mpl = mp.strip().lower()
                # match if either string is a substring of the other (handles variants)
                if ml in mpl or mpl in ml:
                    matched = True
                    break
            if matched:
                kept_markets.append(m)
        kept_crops = [c for c in info.get('crops', []) if crop_present_in_data(loader, c, varieties)]
        # Only include district if any markets/crops remain
        if kept_markets or kept_crops:
            suggested[district] = {
                'markets': kept_markets,
                'crops': kept_crops
            }

    print("Suggested DISTRICTS_DATA mapping (JSON):")
    print(json.dumps(suggested, indent=2, ensure_ascii=False))

    # Also print a Python literal for easy copy/paste
    print('\nSuggested Python dict literal:')
    print(repr(suggested))


if __name__ == '__main__':
    main()
