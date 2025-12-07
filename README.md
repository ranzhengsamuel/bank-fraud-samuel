# Bank Fraud Detection

Machine learning project for detecting fraudulent bank transactions.

## Dataset

The dataset files are **not included** in this repository due to their large size (>100MB each):
- `data/Base.csv` (203.54 MB) - Original dataset
- `data/preprocessed.csv` (164.16 MB) - Preprocessed dataset

### Getting the Data

**Option 1: Download from source**
- Place `Base.csv` in the `data/` directory
- Run the preprocessing script to generate `preprocessed.csv`

**Option 2: Run preprocessing yourself**
```bash
# Place Base.csv in data/ directory first
python preprocess.py
```

This will generate:
- `data/preprocessed.csv` - Cleaned and processed dataset
- `data/encoding_mappings.json` - Categorical encoding reference

## Project Structure

```
bank-fraud/
├── data/
│   ├── Base.csv              # Original dataset (NOT in git)
│   ├── preprocessed.csv      # Processed dataset (NOT in git)
│   └── encoding_mappings.json # Encoding mappings (in git)
├── preprocess.py             # Data preprocessing script
├── preprocess.ipynb          # Jupyter notebook version
├── ENCODING_REFERENCE.md     # Encoding and transformation guide
└── README.md                 # This file
```

## Preprocessing

The preprocessing script performs:

1. **Data type optimization** - Reduces memory usage by 71.8%
2. **Category consolidation** - Reduces model parameters from 22+ to 16 categories
3. **Label encoding** - Encodes categorical variables for ML models
4. **Feature engineering** - Creates 5 new derived features
5. **Missing value handling** - Imputes missing values appropriately

### Key Transformations

- **Employment Status**: 7 categories → 3 (Employed_A, Employed_B, Other)
- **Housing Status**: 7 categories → 2 (Owned_or_Rented, Other)
- **Device OS**: 5 categories → 4 (linux, windows, macos, other)
- **Memory**: 496.60 MB → 140.19 MB (71.8% reduction)

### New Engineered Features

1. `address_stability` - Residential stability metric
2. `velocity_ratio_6h_4w` - Transaction velocity ratio
3. `banking_history_score` - Banking relationship depth
4. `credit_to_income_ratio` - Credit limit vs income
5. `device_fraud_rate` - Device-associated fraud rate

## Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn
```

### Run Preprocessing

```bash
python preprocess.py
```

### Analyze Processed Data

```bash
python analyze_preprocessed.py
```

## Dataset Information

- **Rows**: 1,000,000
- **Features**: 36 (after preprocessing)
- **Target**: `fraud_bool` (binary: 0 = legitimate, 1 = fraud)
- **Class Distribution**: 1.10% fraud (highly imbalanced - 1:89.7 ratio)

## Documentation

- **ENCODING_REFERENCE.md** - Detailed encoding and consolidation reference
- **data/encoding_mappings.json** - JSON mapping of categorical encodings

## Next Steps

1. Split data into train/validation/test sets (use stratified split)
2. Handle class imbalance (SMOTE, class weights, or ensemble methods)
3. Train baseline models (XGBoost, LightGBM recommended for tree-based)
4. Evaluate using precision, recall, F1-score, ROC-AUC (not accuracy)
5. Perform feature selection and importance analysis

## Notes

- Large CSV files are excluded via `.gitignore` to comply with GitHub's 100MB limit
- Consider using Git LFS if you need to version control large data files
- The preprocessing is optimized for tree-based models (XGBoost, Random Forest)
- For neural networks, consider one-hot encoding instead of label encoding
