import os
import h5py

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_models')

def inspect_h5(path):
    info = {}
    try:
        with h5py.File(path, 'r') as f:
            # common attributes
            for a in ['keras_version', 'training_config', 'backend']:
                if a in f.attrs:
                    info[a] = f.attrs[a]
            # some files store model_config as JSON in attributes or as dataset
            if 'model_config' in f.attrs:
                info['model_config'] = f.attrs['model_config']
    except Exception as e:
        info['error'] = str(e)
    return info

def main(limit=20):
    if not os.path.isdir(MODEL_DIR):
        print('trained_models/ missing')
        return
    h5files = [f for f in os.listdir(MODEL_DIR) if f.lower().endswith('.h5')]
    if not h5files:
        print('No .h5 files found')
        return
    print(f'Inspecting {min(len(h5files), limit)} of {len(h5files)} .h5 files')
    stats = {}
    for i, fn in enumerate(sorted(h5files)[:limit], 1):
        path = os.path.join(MODEL_DIR, fn)
        info = inspect_h5(path)
        print(f'[{i}] {fn}:')
        if 'error' in info:
            print('   ERROR:', info['error'])
            continue
        for k, v in info.items():
            if isinstance(v, bytes):
                try:
                    v = v.decode('utf-8')
                except Exception:
                    pass
            # shorten long training_config
            s = v
            if isinstance(s, str) and len(s) > 200:
                s = s[:200] + '...'
            print(f'   {k}: {s}')
            stats.setdefault(k, {})
            stats[k][v] = stats[k].get(v, 0) + 1
    print('\nSummary stats:')
    for k, m in stats.items():
        print(f' {k}:')
        for val, cnt in sorted(m.items(), key=lambda x: -x[1]):
            v = val
            if isinstance(v, bytes):
                try:
                    v = v.decode('utf-8')
                except Exception:
                    pass
            print(f'   {cnt}x {v}')

if __name__ == '__main__':
    main()
