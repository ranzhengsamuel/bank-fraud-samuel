# Encoding Reference Guide

## Quick Summary of Changes

### Category Consolidation Impact
- **Total unique categories**: Reduced from 22+ to 16
- **Model parameters saved**: ~6 fewer categories = simpler, faster models
- **Memory reduction**: 71.8% (496.60 MB → 140.19 MB)

---

## Categorical Variable Mappings

### 1. Payment Type (payment_type_encoded)
**No consolidation** - All 5 categories kept as distinct
```
AA → 0  (258,249 records - 25.8%)
AB → 1  (370,554 records - 37.1%)
AC → 2  (252,071 records - 25.2%)
AD → 3  (118,837 records - 11.9%)
AE → 4  (289 records - 0.03%)
```

### 2. Employment Status (employment_status_encoded)
**Consolidated from 7 to 3 categories**
```
BEFORE           AFTER           ENCODED
------           -----           -------
CA        →   Employed_A    →      0     (730,252 records - 73.0%)
CB        →   Employed_B    →      1     (138,288 records - 13.8%)
CC, CD,   →   Other         →      2     (131,460 records - 13.1%)
CE, CF, CG
```

**Rationale**: Categories CC-CG were sparse (<5% each), consolidating reduces overfitting risk

### 3. Housing Status (housing_status_encoded)
**Consolidated from 7 to 2 categories**
```
BEFORE        AFTER              ENCODED
------        -----              -------
BA, BB, BC →  Owned_or_Rented →    1     (802,783 records - 80.3%)
BD, BE,    →  Other           →    0     (197,217 records - 19.7%)
BF, BG
```

**Rationale**: Most records fell into BA/BB/BC (similar housing situations), others are rare

### 4. Source (source_encoded)
**Consolidated from 2 to 2 categories (renamed)**
```
BEFORE       AFTER       ENCODED
------       -----       -------
INTERNET  →  INTERNET →    0     (992,952 records - 99.3%)
TELEAPP   →  OTHER    →    1     (7,048 records - 0.7%)
```

**Rationale**: TELEAPP is very rare, generalized name for extensibility

### 5. Device OS (device_os_encoded)
**Consolidated from 5 to 4 categories**
```
BEFORE        AFTER      ENCODED
------        -----      -------
linux      →  linux   →    0     (332,712 records - 33.3%)
macintosh  →  macos   →    1     (53,826 records - 5.4%)
other      →  other   →    2     (349,956 records - 35.0%)
windows    →  windows →    3     (263,506 records - 26.4%)
x11        →  other   →    2     (merged - x11 is Unix/Linux variant)
```

**Rationale**: x11 is essentially a Unix/Linux variant, merged with "other"

---

## Data Type Optimizations

### Before → After
- **int64** → **int8** for binary (0/1) columns (87.5% savings)
- **int64** → **int32** for count columns (50% savings)
- **float64** → **float32** for continuous variables (50% savings)

### Specific Columns by Type

**Binary (int8) - 7 columns:**
- fraud_bool, email_is_free, phone_home_valid, phone_mobile_valid
- has_other_cards, foreign_request, keep_alive_session

**Count (int32) - 10 columns:**
- prev_address_months_count, current_address_months_count, customer_age
- zip_count_4w, bank_branch_count_8w, date_of_birth_distinct_emails_4w
- bank_months_count, device_distinct_emails_8w, device_fraud_count, month

**Float (float32) - 10 columns:**
- income, name_email_similarity, days_since_request, intended_balcon_amount
- velocity_6h, velocity_24h, velocity_4w, credit_risk_score
- proposed_credit_limit, session_length_in_minutes

**Encoded Categorical (int64) - 5 columns:**
- payment_type_encoded, employment_status_encoded, housing_status_encoded
- source_encoded, device_os_encoded

**Engineered (float32) - 5 columns:**
- address_stability, velocity_ratio_6h_4w, banking_history_score
- credit_to_income_ratio, device_fraud_rate

---

## Feature Engineering Details

### 1. address_stability
```python
current_address_months / (current_address_months + prev_address_months + 1)
```
- Range: [0, 1]
- Higher value = more stable address history
- Missing: 241 records (0.024%)

### 2. velocity_ratio_6h_4w
```python
velocity_6h / (velocity_4w + 1)
```
- Compares short-term vs long-term transaction velocity
- High ratio may indicate unusual activity burst

### 3. banking_history_score
```python
bank_months_count × bank_branch_count_8w
```
- Combined metric of banking relationship depth
- Higher = longer history × more branches

### 4. credit_to_income_ratio
```python
proposed_credit_limit / (income + 0.1)
```
- Credit limit relative to income
- High ratio = high credit relative to income

### 5. device_fraud_rate
```python
device_fraud_count / (device_distinct_emails_8w + 1)
```
- Fraud rate associated with device
- Missing: 359 records (0.036%)
- Range: [0, 1]

---

## Missing Value Handling

### Values Imputed:
- **prev_address_months_count**: 712,920 records had -1 → replaced with median (34.0)
- **bank_months_count**: 253,635 records had -1 → replaced with median (15.0)

### Remaining Missing Values:
- **address_stability**: 241 NaN (from division edge cases)
- **device_fraud_rate**: 359 NaN (from division edge cases)

**Note**: Other columns with -1 values were kept as-is, as -1 may be a meaningful indicator (e.g., "not applicable")

---

## Model Recommendations

### For Tree-Based Models (XGBoost, LightGBM, Random Forest):
✅ Use as-is with label encoding
✅ Models handle categorical encoded values well
✅ No need for additional scaling

### For Linear Models (Logistic Regression, SVM):
⚠️ Consider one-hot encoding instead of label encoding
⚠️ Apply feature scaling (StandardScaler or MinMaxScaler)
⚠️ Handle class imbalance (1:89.7 ratio)

### For Neural Networks:
⚠️ Use one-hot encoding for categorical variables
⚠️ Apply normalization/standardization
⚠️ Consider embedding layers for high-cardinality categoricals

---

## Class Imbalance

**Current distribution:**
- Legitimate: 988,971 (98.90%)
- Fraudulent: 11,029 (1.10%)
- Ratio: 1:89.7

**Recommended strategies:**
1. Use stratified train/test split
2. Apply SMOTE or ADASYN for oversampling
3. Use class weights in model training
4. Use ensemble methods designed for imbalanced data
5. Evaluate with precision, recall, F1, ROC-AUC (not accuracy)

---

## Files Generated

1. **data/preprocessed.csv** - Ready-to-use dataset (282 MB)
2. **data/encoding_mappings.json** - Categorical encoding reference
3. **preprocess.py** - Reproducible preprocessing script
4. **analyze_preprocessed.py** - Data analysis script
5. **PREPROCESSING_SUMMARY.md** - Detailed documentation
6. **ENCODING_REFERENCE.md** - This file
