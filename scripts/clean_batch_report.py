import json
from pathlib import Path

REPORT = Path('batch_train_report.json')
OUT = Path('batch_train_report_cleaned.json')

if not REPORT.exists():
    raise SystemExit('batch_train_report.json not found')

with REPORT.open('r', encoding='utf-8') as f:
    report = json.load(f)

skipped = report.get('skipped', [])
candidates = report.get('candidates', {})

# reasons that indicate crop should be removed from candidates
BAD = {'no_data', 'insufficient_numeric_samples', 'no_price_column', 'insufficient_data_after_prepare'}

bad_set = set()
for s in skipped:
    if s.get('reason') in BAD:
        bad_set.add((s.get('district'), s.get('market'), s.get('crop')))

cleaned = {}
for dist, info in candidates.items():
    markets = info.get('markets', [])
    crops = info.get('crops', [])
    kept_markets = []
    kept_crops = []
    # keep only crops that don't appear in bad_set for any market in this district
    for m in markets:
        # determine which crops for this market should be kept
        m_kept = [c for c in crops if (dist, m, c) not in bad_set]
        if m_kept:
            kept_markets.append(m)
            # accumulate crops (union across markets)
            for c in m_kept:
                if c not in kept_crops:
                    kept_crops.append(c)
    if kept_markets and kept_crops:
        cleaned[dist] = {'markets': kept_markets, 'crops': sorted(kept_crops)}

report['candidates_cleaned'] = cleaned

with OUT.open('w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print('WROTE', OUT)
