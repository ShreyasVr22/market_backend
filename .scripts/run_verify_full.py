import importlib.util
from pathlib import Path

script_path = Path(__file__).parent / 'verify_models.py'
spec = importlib.util.spec_from_file_location('verify_models', str(script_path))
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)

if __name__ == '__main__':
    # increase limit so all models are checked
    vm.main(limit=10000)
