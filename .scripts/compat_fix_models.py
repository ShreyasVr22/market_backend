import os
import json
import shutil
import tempfile
import h5py
import traceback

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'trained_models')
OUT_DIR = os.path.join(ROOT, 'trained_models_compatible')

def remove_time_major(obj):
    if isinstance(obj, dict):
        obj.pop('time_major', None)
        for k, v in list(obj.items()):
            remove_time_major(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_time_major(item)

def copy_with_fixed_model_config(src, dst):
    # copy whole file then overwrite model_config attr if present
    with h5py.File(src, 'r') as fr:
        model_config = None
        if 'model_config' in fr.attrs:
            raw = fr.attrs['model_config']
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                cfg = json.loads(raw)
                remove_time_major(cfg)
                model_config = json.dumps(cfg)
            except Exception:
                model_config = None

    # copy file to dst
    shutil.copy2(src, dst)
    if model_config is not None:
        # write updated attr
        with h5py.File(dst, 'r+') as fw:
            try:
                fw.attrs['model_config'] = model_config
            except Exception:
                # some files may store differently; ignore
                pass

def try_load_model(path):
    try:
        # attempt to load using Keras
        from tensorflow.keras.models import load_model
    except Exception:
        try:
            from keras.models import load_model
        except Exception:
            return False, 'no-load_model-available'

    try:
        m = load_model(path)
        return True, None
    except Exception as e:
        return False, str(e)

def main(limit=100):
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.lower().endswith('.h5')])
    if not files:
        print('No .h5 models found in', MODEL_DIR)
        return

    inspected = 0
    success = 0
    failed = 0
    for fn in files[:limit]:
        inspected += 1
        src = os.path.join(MODEL_DIR, fn)
        dst = os.path.join(OUT_DIR, fn)
        print(f'[{inspected}] Fixing {fn} -> trained_models_compatible/{fn} ...', end=' ')
        try:
            copy_with_fixed_model_config(src, dst)
            ok, err = try_load_model(dst)
            if ok:
                print('LOAD OK')
                success += 1
            else:
                print('LOAD FAILED:', err)
                failed += 1
        except Exception:
            print('ERROR')
            traceback.print_exc()
            failed += 1

    print('\nDone: inspected', inspected, 'success', success, 'failed', failed)

if __name__ == '__main__':
    main()
