# Starts the FastAPI app using the TF 2.14 disposable venv
$venvPython = Join-Path $PSScriptRoot "..\venv_tf214\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Error "Python not found at $venvPython. Ensure .venv_tf214 exists."
    exit 1
}

Write-Output "Starting server with: $venvPython -m uvicorn main:app --host 127.0.0.1 --port 8001"
& $venvPython -m uvicorn main:app --host 127.0.0.1 --port 8001
