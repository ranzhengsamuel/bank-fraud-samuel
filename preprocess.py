import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(input_file='data/Base.csv', output_file='data/preprocessed.csv'):
    """
    Preprocess bank fraud dataset with proper data types, encoding, and category consolidation.
    """
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Dataset shape: {df.shape}")
    print(f"\nOriginal columns and types:")
    print(df.dtypes)

    # Create a copy for preprocessing
    df_processed = df.copy()

    # ==============================================
    # 1. DATA TYPE CONVERSIONS
    # ==============================================
    print("\n" + "="*50)
    print("STEP 1: Converting to proper data types")
    print("="*50)

    # Binary columns (should be int)
    binary_cols = [
        'fraud_bool',           # Target variable
        'email_is_free',        # Boolean flags
        'phone_home_valid',
        'phone_mobile_valid',
        'has_other_cards',
        'foreign_request',
        'keep_alive_session'
    ]

    for col in binary_cols:
        df_processed[col] = df_processed[col].astype('int8')

    # Count/integer columns
    count_cols = [
        'prev_address_months_count',
        'current_address_months_count',
        'customer_age',
        'zip_count_4w',
        'bank_branch_count_8w',
        'date_of_birth_distinct_emails_4w',
        'bank_months_count',
        'device_distinct_emails_8w',
        'device_fraud_count',
        'month'
    ]

    for col in count_cols:
        df_processed[col] = df_processed[col].astype('int32')

    # Float columns (numerical features)
    float_cols = [
        'income',
        'name_email_similarity',
        'days_since_request',
        'intended_balcon_amount',
        'velocity_6h',
        'velocity_24h',
        'velocity_4w',
        'credit_risk_score',
        'proposed_credit_limit',
        'session_length_in_minutes'
    ]

    for col in float_cols:
        df_processed[col] = df_processed[col].astype('float32')

    # Categorical columns (will be encoded)
    categorical_cols = [
        'payment_type',
        'employment_status',
        'housing_status',
        'source',
        'device_os'
    ]

    print(f"Binary columns: {len(binary_cols)}")
    print(f"Count columns: {len(count_cols)}")
    print(f"Float columns: {len(float_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # ==============================================
    # 2. CATEGORY CONSOLIDATION
    # ==============================================
    print("\n" + "="*50)
    print("STEP 2: Consolidating categories")
    print("="*50)

    # Analyze and consolidate payment_type
    print("\nPayment Type distribution:")
    print(df_processed['payment_type'].value_counts())
    # Keep as is - only 4 categories (AA, AB, AC, AD)

    # Analyze and consolidate employment_status
    print("\nEmployment Status distribution:")
    print(df_processed['employment_status'].value_counts())
    # Consolidate employment status - combine rare categories
    employment_mapping = {
        'CA': 'Employed_A',      # Most common
        'CB': 'Employed_B',
        'CC': 'Other',           # Consolidate rare
        'CD': 'Other',
        'CE': 'Other',
        'CF': 'Other'
    }
    df_processed['employment_status'] = df_processed['employment_status'].map(
        lambda x: employment_mapping.get(x, 'Other')
    )
    print("\nConsolidated Employment Status:")
    print(df_processed['employment_status'].value_counts())

    # Analyze and consolidate housing_status
    print("\nHousing Status distribution:")
    print(df_processed['housing_status'].value_counts())
    # Consolidate housing status - combine similar categories
    housing_mapping = {
        'BC': 'Owned_or_Rented',     # Most common - combine
        'BA': 'Owned_or_Rented',
        'BB': 'Owned_or_Rented',
        'BD': 'Other',               # Less common
        'BE': 'Other',
        'BF': 'Other'
    }
    df_processed['housing_status'] = df_processed['housing_status'].map(
        lambda x: housing_mapping.get(x, 'Other')
    )
    print("\nConsolidated Housing Status:")
    print(df_processed['housing_status'].value_counts())

    # Analyze and consolidate source
    print("\nSource distribution:")
    print(df_processed['source'].value_counts())
    # Keep INTERNET as main, consolidate others
    df_processed['source'] = df_processed['source'].apply(
        lambda x: 'INTERNET' if x == 'INTERNET' else 'OTHER'
    )
    print("\nConsolidated Source:")
    print(df_processed['source'].value_counts())

    # Analyze and consolidate device_os
    print("\nDevice OS distribution:")
    print(df_processed['device_os'].value_counts())
    # Consolidate device OS - keep major ones, combine rare
    device_os_mapping = {
        'windows': 'windows',
        'linux': 'linux',
        'other': 'other',
        'macintosh': 'macos',
        'x11': 'other'           # X11 is essentially Unix/Linux variant
    }
    df_processed['device_os'] = df_processed['device_os'].map(
        lambda x: device_os_mapping.get(x, 'other')
    )
    print("\nConsolidated Device OS:")
    print(df_processed['device_os'].value_counts())

    # ==============================================
    # 3. ENCODING CATEGORICAL VARIABLES
    # ==============================================
    print("\n" + "="*50)
    print("STEP 3: Encoding categorical variables")
    print("="*50)

    # Use Label Encoding for categorical variables
    # This is appropriate for tree-based models and reduces dimensionality
    label_encoders = {}

    for col in categorical_cols:
        print(f"\nEncoding {col}:")
        print(f"  Unique values: {df_processed[col].nunique()}")
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  Encoding mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        # Keep original column for reference but will use encoded version

    # ==============================================
    # 4. FEATURE ENGINEERING
    # ==============================================
    print("\n" + "="*50)
    print("STEP 4: Feature engineering")
    print("="*50)

    # Create useful derived features

    # Address stability ratio
    df_processed['address_stability'] = (
        df_processed['current_address_months_count'] /
        (df_processed['current_address_months_count'] + df_processed['prev_address_months_count'] + 1)
    ).astype('float32')

    # Velocity ratio (short term vs long term)
    df_processed['velocity_ratio_6h_4w'] = (
        df_processed['velocity_6h'] / (df_processed['velocity_4w'] + 1)
    ).astype('float32')

    # Banking history score
    df_processed['banking_history_score'] = (
        df_processed['bank_months_count'] * df_processed['bank_branch_count_8w']
    ).astype('float32')

    # Credit limit to income ratio
    df_processed['credit_to_income_ratio'] = (
        df_processed['proposed_credit_limit'] / (df_processed['income'] + 0.1)
    ).astype('float32')

    # Device fraud rate
    df_processed['device_fraud_rate'] = (
        df_processed['device_fraud_count'] / (df_processed['device_distinct_emails_8w'] + 1)
    ).astype('float32')

    print("Created 5 new engineered features")

    # ==============================================
    # 5. HANDLE MISSING/INVALID VALUES
    # ==============================================
    print("\n" + "="*50)
    print("STEP 5: Handling missing/invalid values")
    print("="*50)

    # Check for -1 values (often used as missing indicator)
    print("\nColumns with -1 values:")
    for col in df_processed.columns:
        if (df_processed[col] == -1).any():
            count = (df_processed[col] == -1).sum()
            print(f"  {col}: {count} ({100*count/len(df_processed):.2f}%)")

    # Replace -1 with median for numerical columns (excluding categorical)
    for col in ['prev_address_months_count', 'bank_months_count']:
        if (df_processed[col] == -1).any():
            median_val = df_processed[df_processed[col] != -1][col].median()
            df_processed[col] = df_processed[col].replace(-1, median_val)
            print(f"Replaced -1 in {col} with median: {median_val}")

    # Check for infinity values
    print("\nChecking for infinity values...")
    inf_cols = df_processed.columns[df_processed.isin([np.inf, -np.inf]).any()].tolist()
    if inf_cols:
        print(f"Found infinity in: {inf_cols}")
        for col in inf_cols:
            df_processed[col] = df_processed[col].replace([np.inf, -np.inf], df_processed[col].median())
    else:
        print("No infinity values found")

    # Check for NaN values
    print("\nChecking for NaN values...")
    nan_counts = df_processed.isnull().sum()
    if nan_counts.any():
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaN values found")

    # ==============================================
    # 6. CREATE FINAL FEATURE SET
    # ==============================================
    print("\n" + "="*50)
    print("STEP 6: Creating final feature set")
    print("="*50)

    # Select final columns (use encoded versions instead of original categorical)
    final_cols = []

    # Target variable
    final_cols.append('fraud_bool')

    # Numerical features
    final_cols.extend(binary_cols[1:])  # Exclude fraud_bool (already added)
    final_cols.extend(count_cols)
    final_cols.extend(float_cols)

    # Encoded categorical features
    for col in categorical_cols:
        final_cols.append(col + '_encoded')

    # Engineered features
    final_cols.extend([
        'address_stability',
        'velocity_ratio_6h_4w',
        'banking_history_score',
        'credit_to_income_ratio',
        'device_fraud_rate'
    ])

    df_final = df_processed[final_cols].copy()

    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"Total features: {df_final.shape[1] - 1} (excluding target)")
    print(f"\nFeature types:")
    print(df_final.dtypes.value_counts())

    # ==============================================
    # 7. DATA QUALITY CHECKS
    # ==============================================
    print("\n" + "="*50)
    print("STEP 7: Data quality checks")
    print("="*50)

    print(f"\nTarget variable distribution:")
    print(df_final['fraud_bool'].value_counts())
    print(f"Fraud rate: {100 * df_final['fraud_bool'].mean():.2f}%")

    print(f"\nMemory usage:")
    print(f"Before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"After optimization: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Basic statistics
    print(f"\nBasic statistics:")
    print(df_final.describe())

    # ==============================================
    # 8. SAVE PREPROCESSED DATA
    # ==============================================
    print("\n" + "="*50)
    print("STEP 8: Saving preprocessed data")
    print("="*50)

    df_final.to_csv(output_file, index=False)
    print(f"\nPreprocessed data saved to: {output_file}")

    # Save encoding mappings for future reference
    import json
    encoding_info = {}
    for col, le in label_encoders.items():
        encoding_info[col] = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))

    with open('data/encoding_mappings.json', 'w') as f:
        json.dump(encoding_info, f, indent=2)
    print("Encoding mappings saved to: data/encoding_mappings.json")

    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)

    return df_final, label_encoders


if __name__ == "__main__":
    df_preprocessed, encoders = preprocess_data()
    print("\nPreprocessing completed successfully!")
    print(f"Final shape: {df_preprocessed.shape}")
    print(f"\nColumn names:")
    for i, col in enumerate(df_preprocessed.columns, 1):
        print(f"  {i}. {col}")
