import traceback, sys
path = r"trained_models\\Bangalore_Bangalore_Tomato.h5"
print('Attempting to load', path)
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    try:
        from keras.models import load_model
    except Exception as e2:
        print('No keras/tensorflow available:', e)
        print('Fallback keras import error:', e2)
        sys.exit(2)
try:
    m = load_model(path)
    print('Loaded OK. Model type:', type(m))
except Exception:
    traceback.print_exc()
    sys.exit(1)
