# PULL Model Improvements for Hetionet

## Overview
This document describes the improvements made to address overfitting, score saturation, and disease bias in the PULL model when applied to the Hetionet dataset.

## Problems Addressed

### 1. Score Saturation
**Problem:** All prediction probabilities converging to 1.0000  
**Solution:** Temperature scaling (T=2.0)

### 2. Disease Bias
**Problem:** Top 20 candidates mostly biased towards hypertension  
**Solution:** Diversity algorithm with max 3 recommendations per disease

### 3. Biological Implausibility
**Problem:** Anticonvulsants recommended as hypertension treatments  
**Solution:** Filter overconfident predictions (>= 0.99) and improve regularization

### 4. Overfitting
**Problem:** Model too confident on training data  
**Solution:** Increased dropout (0.2 → 0.3), reduced learning rate (0.01 → 0.005), reduced PULL ratio (0.05 → 0.03)

## Changes Made

### Model Architecture (`src/model_hetionet.py`)
1. **Temperature Scaling**
   - Added `temperature` parameter (default=2.0)
   - Applied in `decode()`: `logits / self.temperature`
   - Applied in `decode_all()`: `raw_scores / self.temperature`

2. **Increased Regularization**
   - Dropout: 0.2 → 0.3

3. **Weight Clipping**
   - PULL edge weights clipped to [0.5, 0.95] range
   - Prevents overconfident predictions

### Training Logic (`src/train_hetionet.py`)
1. **Reduced PULL Ratio**
   - Changed from 0.05 to 0.03
   - Less aggressive edge addition

2. **Weight Clipping**
   - Applied to newly added PULL edges
   - Range: [0.5, 0.95]

3. **Diversity Algorithm**
   - Initial candidate pool: 10x requested (200 for top 20)
   - Filter predictions with probability >= 0.99
   - Limit to max 3 recommendations per disease
   - Sort by raw score (descending)

### Main Script (`main_hetionet.py`)
1. **Reduced Learning Rate**
   - Changed from 0.01 to 0.005
   - More stable training

2. **New CLI Argument**
   - `--temperature`: Configure temperature scaling (default=2.0)

## Usage

### Default Settings (Recommended)
```bash
python main_hetionet.py
```

### Custom Temperature
```bash
python main_hetionet.py --temperature 3.0
```

### Custom Learning Rate
```bash
python main_hetionet.py --lr 0.001
```

## Expected Results

### Before Changes
- Prediction probabilities: all ~1.0000
- Top 20 candidates: 15+ hypertension recommendations
- Low diversity in disease types
- Biologically implausible recommendations

### After Changes
- Prediction probabilities: distributed range (e.g., 0.70-0.98)
- Top 20 candidates: max 3 per disease type
- High diversity across different diseases
- More biologically plausible recommendations
- Better generalization performance

## Technical Details

### Temperature Scaling Formula
```python
# Before
logits = (compound_embeds * disease_embeds).sum(dim=-1)
prob = sigmoid(logits)

# After
logits = (compound_embeds * disease_embeds).sum(dim=-1)
logits = logits / temperature  # Reduces overconfidence
prob = sigmoid(logits)
```

### Weight Clipping Formula
```python
# Before
edge_weight = sigmoid(score)  # Can be 0.999+

# After
edge_weight = sigmoid(score)
edge_weight = clamp(edge_weight, min=0.5, max=0.95)  # Bounded
```

### Diversity Algorithm
```python
for each candidate in top_200:
    if prob >= 0.99: skip  # Too confident
    if disease_count[disease] >= 3: skip  # Too many for this disease
    add to final_candidates
    if len(final_candidates) >= 20: break
```

## Configuration Summary

| Parameter | Before | After | Purpose |
|-----------|--------|-------|---------|
| Temperature | - | 2.0 | Prevent saturation |
| Dropout | 0.2 | 0.3 | Reduce overfitting |
| Learning Rate | 0.01 | 0.005 | Stable training |
| PULL Ratio | 0.05 | 0.03 | Conservative edge addition |
| Weight Range | [0, 1] | [0.5, 0.95] | Prevent overconfidence |
| Max per Disease | - | 3 | Ensure diversity |
| Prob Filter | - | < 0.99 | Remove overconfident |

## Validation

Run the verification script to ensure all changes are present:
```bash
python /tmp/verify_changes.py
```

All checks should pass with "✅ ALL CHECKS PASSED!" message.
