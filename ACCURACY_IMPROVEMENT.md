# Accuracy Improvement Strategy

## Current Status
- **Models trained**: 191 LSTM models (1 epoch, batch_size 8)
- **Re-training in progress**: 50 epochs, batch_size 16 to improve accuracy

## Why 50 Epochs Improves Accuracy

1. **Better Convergence**: LSTM models with only 1 epoch rarely reach optimal weights. 50 epochs allows the model to:
   - Reduce training loss significantly
   - Learn complex patterns in price time-series
   - Stabilize predictions

2. **Batch Size Optimization**: 
   - Original: batch_size=8 (too small, noisy gradients)
   - New: batch_size=16 (better gradient estimates, faster convergence)

3. **Typical Improvement Expected**:
   - Loss reduction: 40-60% lower on validation
   - RMSE (Root Mean Square Error): 15-25% improvement in out-of-sample predictions

## Implementation Details

- **Script**: `scripts/retrain_for_accuracy.py`
- **Backup**: Old models backed up to `trained_models_backup_TIMESTAMP`
- **Output**: New models in `trained_models/` with improved weights
- **Log**: Training report saved to `logs/retrain_TIMESTAMP.json`

## Usage

```powershell
# Re-train with 50 epochs (default)
.venv\Scripts\python.exe scripts\retrain_for_accuracy.py

# Custom epochs
.venv\Scripts\python.exe scripts\retrain_for_accuracy.py --epochs 100

# Custom batch size
.venv\Scripts\python.exe scripts\retrain_for_accuracy.py --epochs 50 --batch_size 32
```

## Expected Outcomes

After re-training completes:
1. All 191 models will have improved weights from 50 epochs
2. Backup available if you need to revert (`trained_models_backup_*`)
3. Price predictions will be more accurate, especially for volatile crops

## Next Steps for Further Accuracy Gains

1. **Data normalization**: Normalize price to 0-1 range per market/crop (already done by scaler)
2. **Feature engineering**: Add moving averages, volatility indicators
3. **Ensemble methods**: Combine LSTM with other models (Random Forest, XGBoost)
4. **Hyperparameter tuning**: Optimize LSTM layers, dropout, learning rate
5. **Expand lookback window**: Currently 30 days; try 60-90 days for seasonality

## Monitoring Accuracy

Check logs after training completes:
```powershell
Get-Content logs/retrain_*.json | ConvertFrom-Json
```

This will show:
- Number of models successfully re-trained
- Any failures or skips
- Total processing time
