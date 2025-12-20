import os
import pickle
import traceback

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_models')

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


def main(limit=30):
    load_model = try_import_load_model()
    if load_model is None:
        print('✗ No Keras/TensorFlow load_model available in environment')

    if not os.path.isdir(MODEL_DIR):
        print(f'No trained_models directory at {MODEL_DIR}')
        return

    files = sorted([f for f in os.listdir(MODEL_DIR) if f.lower().endswith('.h5') or f.lower().endswith('.pkl')])
    if not files:
        print('No model files found in trained_models/')
        return

    print(f'Found {len(files)} model files (showing up to {limit})')
    ok = 0
    failed = 0
    for i, fname in enumerate(files[:limit], 1):
        path = os.path.join(MODEL_DIR, fname)
        print(f'[{i}] Checking {fname}...', end=' ')
        try:
            if fname.lower().endswith('.h5'):
                if load_model is None:
                    raise RuntimeError('load_model not available')
                m = load_model(path)
                print('OK (h5 loaded)')
                # try scaler
                scaler_path = os.path.splitext(path)[0] + '_scaler.pkl'
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as sf:
                            s = pickle.load(sf)
                        print('    scaler loaded')
                    except Exception as se:
                        print('    scaler load FAILED:', se)
                ok += 1
            else:
                # pickle
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                print('OK (pickle loaded)')
                # if dict, check model/scaler keys
                if isinstance(obj, dict):
                    print('    keys:', list(obj.keys()))
                ok += 1
        except Exception as e:
            print('FAILED')
            traceback.print_exc()
            failed += 1

    print('\nSummary:')
    print('  total inspected:', min(len(files), limit))
    print('  ok:', ok)
    print('  failed:', failed)

if __name__ == "__main__":
    main()
