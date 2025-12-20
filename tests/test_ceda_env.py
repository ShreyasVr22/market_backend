import os
import sys
import pytest

# Basic check: if user has not provided CEDA_API_KEY in .env, skip this test.
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def test_ceda_api_key_present():
    key = os.getenv('CEDA_API_KEY')
    if not key:
        pytest.skip('CEDA_API_KEY not set in environment; skipping integration test')
    assert len(key) > 10, 'CEDA_API_KEY looks too short'
