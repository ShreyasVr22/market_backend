import os
import json
from collections import defaultdict
import pandas as pd

from models.data_loader import DataLoader
from models.trainer import train_and_save_all


def filter_candidates(loader, candidates, min_samples=60):
    """Validate and return a reduced candidates dict keeping only (market,crop)
    combos that actually have >= min_samples numeric price samples.
    """
    out = defaultdict(lambda: {'markets': set(), 'crops': set()})
    price_col = 'Modal Price (Rs./Quintal)'
    for dist, info in candidates.items():
        for market in info.get('markets', []):
            for crop in info.get('crops', []):
                df_check = loader.filter_by_market_crop(market, crop)
                if df_check is None or df_check.empty:
                    continue
                if price_col not in df_check.columns:
                    continue
                numeric_prices = pd.to_numeric(df_check[price_col], errors='coerce').dropna()
                if len(numeric_prices) >= min_samples:
                    out[dist]['markets'].add(market)
                    out[dist]['crops'].add(crop)

    cleaned = {}
    for dist, info in out.items():
        cleaned[dist] = {
            'markets': sorted(list(info['markets'])),
            'crops': sorted(list(info['crops']))
        }
    return cleaned

def collect_candidates(loader, min_samples=60):
    """Return a districts_data dict containing only market/crop combos
    that have at least `min_samples` numeric price entries.
    """
    if loader.data is None:
        loader.load_csv()
    df = loader.data
    if df is None:
        raise SystemExit('No CSV loaded')

    # determine district column
    district_cols = ['District', 'District Name', 'District_Name', 'district', 'district_name']
    district_col = next((c for c in district_cols if c in df.columns), None)

    # gather crop columns
    crop_cols = [c for c in ['Variety', 'Commodity', 'Group'] if c in df.columns]
    price_col = 'Modal Price (Rs./Quintal)'

    candidates = defaultdict(lambda: {'markets': set(), 'crops': set()})

    # group by district and market to count per-crop
    for idx, row in df.iterrows():
        # determine district key
        if district_col:
            dist = str(row.get(district_col, '')).strip()
        else:
            # infer from Market Name first token
            market_name = str(row.get('Market Name', '')).strip()
            dist = market_name.split('_')[0].split('-')[0].split()[0] if market_name else 'Unknown'

        market = str(row.get('Market Name', '')).strip()
        # find crop values from crop_cols
        for cc in crop_cols:
            crop = str(row.get(cc, '')).strip()
            if not crop:
                continue
            # count numeric price
            try:
                val = float(str(row.get(price_col, '')).replace(',', '').replace('Rs.', '').replace('₹', ''))
            except Exception:
                continue
            # increment count for this (dist, market, crop)
            key = (dist, market, crop)
            # we'll accumulate counts in a dict later; for now, record existence
            candidates[(dist, market)]['crops'].add(crop)
            candidates[(dist, market)]['markets'].add(market)

    # Now compute numeric counts robustly using groupby
    counts = {}
    if price_col in df.columns and crop_cols:
        df_price = df.copy()
        df_price[price_col] = df_price[price_col].astype(str).str.replace(',', '').str.replace('Rs.', '').str.replace('₹', '')
        df_price[price_col] = df_price[price_col].apply(lambda x: float(x) if x.replace('.', '', 1).isdigit() else None)
        df_price = df_price.dropna(subset=[price_col])

        # iterate districts/markets from candidates
        out = defaultdict(lambda: {'markets': set(), 'crops': set()})
        for (dist, market), info in list(candidates.items()):
            market_mask = df_price['Market Name'].astype(str).str.strip().eq(market)
            if district_col:
                dist_mask = df_price[district_col].astype(str).str.strip().eq(dist)
                mask = market_mask & dist_mask
            else:
                mask = market_mask
            if not mask.any():
                continue
            sub = df_price[mask]
            # count per crop across crop_cols
            crop_counts = {}
            for cc in crop_cols:
                if cc not in sub.columns:
                    continue
                g = sub.groupby(cc)[price_col].count()
                for crop, cnt in g.items():
                    key = str(crop).strip()
                    crop_counts[key] = crop_counts.get(key, 0) + int(cnt)

            # select crops meeting threshold
            selected = [c for c, cnt in crop_counts.items() if cnt >= min_samples]
            if selected:
                out[dist]['markets'].add(market)
                out[dist]['crops'].update(selected)

        # convert sets to lists
        districts_data = {}
        for dist, info in out.items():
            districts_data[dist] = {
                'markets': sorted(list(info['markets'])),
                'crops': sorted(list(info['crops']))
            }
        return districts_data

    return {}


def main():
    csv_path = os.environ.get('CSV_PATH')
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit('CSV_PATH not set or file missing')

    min_samples = int(os.environ.get('TRAIN_MIN_SAMPLES', '60'))
    epochs = int(os.environ.get('TRAIN_EPOCHS', '5'))
    batch_size = int(os.environ.get('TRAIN_BATCH_SIZE', '16'))

    loader = DataLoader(csv_path)
    loader.load_csv()

    candidates = collect_candidates(loader, min_samples=min_samples)
    report_path = os.environ.get('BATCH_TRAIN_REPORT', 'batch_train_report.json')

    if not candidates:
        print('No eligible market/crop combos found for min_samples=', min_samples)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({'trained': [], 'skipped': [], 'failed': [], 'candidates': {}}, f, ensure_ascii=False, indent=2)
        return

    print('Candidates districts count (raw):', len(candidates))

    # Filter candidates to ensure they truly have enough numeric samples
    cleaned_candidates = filter_candidates(loader, candidates, min_samples=min_samples)
    print('Candidates districts count (cleaned):', len(cleaned_candidates))

    if not cleaned_candidates:
        print('No cleaned candidates after validation')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({'trained': [], 'skipped': [], 'failed': [], 'candidates': candidates, 'candidates_cleaned': {}}, f, ensure_ascii=False, indent=2)
        return

    # call trainer on cleaned candidates
    try:
        report = train_and_save_all(loader, cleaned_candidates, epochs=epochs, batch_size=batch_size)
    except Exception as e:
        report = {'trained': [], 'skipped': [], 'failed': [{'error': str(e)}], 'candidates': cleaned_candidates}

    # attach both candidate lists and write report
    report['candidates'] = candidates
    report['candidates_cleaned'] = cleaned_candidates
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('WROTE', report_path)


if __name__ == '__main__':
    main()
