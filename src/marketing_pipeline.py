import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from lightgbm import LGBMClassifier

def load_and_split_data(filepath='../data/raw/online_shoppers_intention.csv'):
    """Loads the dataset and separates the target variable."""
    df = pd.read_csv(filepath)
    X = df.drop('Revenue', axis=1)
    y = df['Revenue'].astype(int)
    return X, y

def build_preprocessor():
    """Builds a 3-pronged preprocessor to handle numeric, low-card, and high-card features."""
    
    # 1. Define our specific feature groups
    numeric_features = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]
    
    # Low cardinality (creates very few new columns)
    low_card_features = ['VisitorType', 'Weekend']
    
    # High cardinality (would create a sparse matrix explosion if we One-Hot Encoded)
    high_card_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']

    # 2. Build the three distinct sub-pipelines
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    low_card_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    high_card_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # TargetEncoder transforms the category into a single numerical probability column
        ('encoder', TargetEncoder(target_type='binary', cv=5)) 
    ])
    
    # 3. Assemble the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat_low', low_card_pipeline, low_card_features),
            ('cat_high', high_card_pipeline, high_card_features)
        ],
        remainder='drop'
    )
    return preprocessor

def build_optimized_lgbm(preprocessor):
    """Assembles the final pipeline using our Optuna-tuned parameters."""
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            class_weight='balanced',
            n_estimators=273,
            learning_rate=0.047897,
            max_depth=11,
            num_leaves=49,
            min_child_samples=12,
            random_state=42,
            verbose=-1  
        ))
    ])
    return model