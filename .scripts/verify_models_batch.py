"""Batch verifier that iterates all files in `trained_models/` and writes
newline-delimited JSON results to `trained_models_verify_tf214_results.ndjson`.
Run this inside the TF 2.14 venv to attempt loading every model safely.
"""
import os
import json
import pickle
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'trained_models'
OUT_FILE = ROOT / 'trained_models_verify_tf214_results.ndjson'


def try_import_load_model():
    try:
        from tensorflow.keras.models import load_model
        return load_model
    except Exception:
        try:
            from keras.models import load_model
            return load_model
        except Exception:
            return None


def main():
    load_model = try_import_load_model()
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.lower().endswith('.h5') or f.lower().endswith('.pkl')])
    total = len(files)
    print(f'Found {total} files')

    with OUT_FILE.open('a', encoding='utf-8') as out:
        for i, fname in enumerate(files, 1):
            path = MODEL_DIR / fname
            res = {'index': i, 'path': str(path), 'status': None, 'note': None, 'error': None}
            try:
                if fname.lower().endswith('.h5'):
                    if load_model is None:
                        raise RuntimeError('load_model not available')
                    _m = load_model(str(path))
                    res['status'] = 'OK'
                    res['note'] = 'h5 loaded'
                    # try scaler
                    scaler_path = path.with_suffix('')
                    scaler_path = Path(str(path).replace('.h5', '_scaler.pkl'))
                    if scaler_path.exists():
                        try:
                            with open(scaler_path, 'rb') as sf:
                                _s = pickle.load(sf)
                            res['note'] += '; scaler loaded'
                        except Exception as se:
                            res['note'] += '; scaler load FAILED'
                else:
                    with open(path, 'rb') as f:
                        obj = pickle.load(f)
                    res['status'] = 'OK'
                    res['note'] = 'pickle loaded'
            except Exception as e:
                res['status'] = 'FAILED'
                res['error'] = ''.join(traceback.format_exception_only(type(e), e)).strip()
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{total}] {fname} -> {res['status']}")


if __name__ == '__main__':
    main()
