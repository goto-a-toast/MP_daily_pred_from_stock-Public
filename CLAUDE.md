# MP Prediction Models Project

## Project Overview

This project contains three machine learning models for predicting MP (Margin Profit/限界利益) using daily data:

1. **Current Month-End MP Prediction Model** - Predicts current month-end MP from daily current month data
2. **Next Month-End MP Prediction Model** - Predicts next month-end MP from daily next month data
3. **Monthly MP Prediction Model** - Predicts next month MP from month-end data

## Models

### 1. Current Month-End MP Prediction (`mp_current_month_prediction_model.ipynb`)

**Purpose**: Predict current month-end MP from daily current month data

**Input Data**:
- 確定MP (Kakutei), AMP (A), BMP (B), CMP (C), DMP (D)
- Columns 2-6 in 13-column format

**Example**:
- Feb 12 data (確定, A, B, C, D) → Predict Feb end MP

**Key Features**:
- Random Forest + Conversion Rate hybrid approach (70% RF + 30% conversion)
- Section 9: Daily prediction progress for latest month
- Section 10: Interactive historical month analysis

### 2. Next Month-End MP Prediction (`mp_next_month_end_prediction_model.ipynb`)

**Purpose**: Predict next month-end MP from daily next month data

**Input Data**:
- 次月確定MP (Next_Kakutei), 次月AMP (Next_A), 次月BMP (Next_B), 次月CMP (Next_C), 次月DMP (Next_D)
- Columns 8-12 in 13-column format

**Example**:
- Feb 12 next month data → Predict Mar end MP

**Key Features**:
- Same logic as current month model but uses next month data
- Section 7: Daily prediction progress for next month-end
- **Section 8: Next Month Start MP Prediction** (IMPORTANT!)
  - Predicts how much "Next Kakutei + Next A" will grow by month-end
  - Uses improved model with Next_B, Next_C, Next_D features
  - Shows current value vs predicted month-end value

### 3. Monthly MP Prediction (`mp_monthly_prediction_model_v4 (1).ipynb`)

**Purpose**: Predict next month MP from month-end data

**Input Data**:
- Month-end values of next month data
- Only uses complete months (reaches actual last day of month)

**Key Features**:
- Monthly simulation for all 12 months
- Month-end filtering using `calendar.monthrange()`

## Data Format

### 13-Column CSV Format (Current)

```csv
列0:  行ラベル (Date)
列1:  合計 / MP (Current Month MP)
列2:  合計 / 確定MP (Current Month Kakutei)
列3:  合計 / AMP (Current Month A)
列4:  合計 / BMP (Current Month B)
列5:  合計 / CMP (Current Month C)
列6:  合計 / DMP (Current Month D)
列7:  合計 / 次月MP (Next Month MP) ⭐NEW
列8:  合計 / 次月確定MP (Next Month Kakutei)
列9:  合計 / 次月AMP (Next Month A)
列10: 合計 / 次月BMP (Next Month B)
列11: 合計 / 次月CMP (Next Month C)
列12: 合計 / 次月DMP (Next Month D)
```

**Backward Compatibility**: Models support 7-column and 12-column formats automatically.

## Model Architecture

All models use a **hybrid approach**:
1. **Random Forest Regressor** (70-80%)
   - Captures complex patterns
   - Features: B, C, D, month, day

2. **Monthly Conversion Rate** (20-30%)
   - Historical average conversion
   - Provides stability

## Development Guidelines

### When Modifying Models

1. **Keep models in sync**: Current month and next month models should use identical logic
2. **Test with multiple data formats**: Ensure backward compatibility with 7, 12, and 13-column formats
3. **Preserve Section 7-8**: These sections visualize predictions for the latest month
4. **Use English labels**: All matplotlib graphs use English to avoid font issues (tofu problem)

### Common Patterns

#### Data Preprocessing
```python
# Get month-end values
month_end_mp = df.groupby('year_month')['MP'].last().to_dict()
df['Target_MP'] = df['year_month'].map(month_end_mp)

# Training data: exclude month-end days
train_data = df[~df['is_month_end']].copy()
```

#### Prediction Function
```python
def predict(self, kakutei, a, b, c, d, month, day):
    confirmed = kakutei + a
    bcd = b + c + d

    # RF prediction
    add_rf = self.model.predict([[b, c, d, month, day]])[0]

    # Conversion rate prediction
    rate = self.monthly_rates.get(month, self.overall_avg_rate) / 100
    add_rate = bcd * rate

    # Hybrid
    additional = add_rf * 0.7 + add_rate * 0.3

    return {
        'forecast': confirmed + additional,
        # ...
    }
```

### Section 8 Special Notes (Next Month Model)

**CRITICAL**: Section 8 predicts month-end growth of "Next Kakutei + Next A"

This is NOT just showing historical values - it's a **prediction model**!

**Example**:
- Feb 12: Next Kakutei + Next A = 4,571 (10K JPY)
- **Prediction**: By Feb 28 → 5,200 (10K JPY)
- **Growth**: +629 (10K JPY)

**Improved Model Features**:
- Uses Next_B, Next_C, Next_D as additional features
- Higher n_estimators (150) and max_depth (10)
- Better R2 score (target: 0.90+)

## Troubleshooting

### Issue: Section 7-8 not working

**Cause**: Data issues or variable scope problems

**Solution**:
- Ensure `df['year_month'].max()` has data
- Check `~df['is_month_end']` condition
- For Section 8, ensure month-end values exist for training

### Issue: Low R2 score

**Symptoms**: R2 < 0.85

**Solutions**:
1. Add more features (B, C, D columns)
2. Increase model complexity (n_estimators, max_depth)
3. Check for data quality issues
4. Review feature correlations

### Issue: Japanese characters show as tofu (□)

**Solution**: All graphs use English labels. If you see Japanese in code:
```python
# Bad
plt.title('予測値の推移')

# Good
plt.title('Prediction Progress')
```

## Git Workflow

### Branch Naming
Use pattern: `claude/predict-monthly-mp-{session-id}`

### Commit Messages
Include session URL at the end:
```
feat: Add improved model for Section 8

- Added Next_B, Next_C, Next_D features
- Improved hyperparameters
- R2 score improved from 0.841 to 0.9XX

https://claude.ai/code/session_XXXXX
```

### Push Commands
Always use retry logic for network issues:
```bash
git push -u origin claude/predict-monthly-mp-XXXXX
# Retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)
```

## Key Files

- `mp_current_month_prediction_model.ipynb` - Current month model
- `mp_next_month_end_prediction_model.ipynb` - Next month model (main focus)
- `mp_monthly_prediction_model_v4 (1).ipynb` - Monthly model
- `README.md` - User documentation with Colab links
- `CLAUDE.md` - This file (developer documentation)

## Performance Targets

| Model | Target R2 | Target MAE |
|-------|-----------|------------|
| Current Month-End | > 0.95 | < 300 (10K JPY) |
| Next Month-End | > 0.90 | < 400 (10K JPY) |
| Section 8 (Next Start MP) | > 0.90 | < 200 (10K JPY) |
| Monthly | > 0.95 | < 300 (10K JPY) |

## Testing Checklist

Before committing changes:
- [ ] Model trains without errors
- [ ] Section 7 displays graph for latest month
- [ ] Section 8 displays prediction (next month model only)
- [ ] Predictions are reasonable (not too high/low)
- [ ] R2 score meets targets
- [ ] All graphs use English labels
- [ ] Backward compatibility works (test with 12-column data)

## Contact & Session

This project was developed with Claude Code assistance.
Session: https://claude.ai/code/session_01PSshXnBStPMdBitsdaBFqQ
