import sys
import os
import pytest

repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi import HTTPException
import routes.market as rm


def test_get_prices_smoke():
    """Smoke test: call the get_prices route function and assert structure.
    If the route raises HTTPException (market/crop not configured), skip the test.
    """
    market = 'Bangalore Central'
    crop = 'Tomato'
    try:
        res = rm.get_prices(market, crop)
    except HTTPException as e:
        pytest.skip(f'get_prices not available for {market}/{crop}: {e.detail}')

    assert isinstance(res, dict)
    assert res.get('market') == market
    assert res.get('crop') == crop
    prices = res.get('prices')
    assert isinstance(prices, dict)
    assert 'current' in prices
    # prediction may be None if model absent; ensure key exists
    assert 'day_30' in prices
