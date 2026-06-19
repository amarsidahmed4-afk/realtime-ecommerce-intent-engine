import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
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

def tune_and_build_pipeline(X_train, y_train, preprocessor, n_trials=30):
    """Runs an Optuna Bayesian Sweep to find the best LightGBM params, then fits the master pipeline."""
    
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        num_leaves = trial.suggest_int("num_leaves", 20, 150)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        
        # Build the cross-validation pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(
                objective='binary',
                class_weight='balanced',
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                colsample_bytree=colsample_bytree,
                verbosity=-1,
                random_state=42
            ))
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring='neg_log_loss', n_jobs=-1)
        return scores.mean()

    print(f"Launching Bayesian Optimization Tournament ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Ultimate Hyperparameters Found:")
    print(study.best_params)

    # Build and fit the final champion pipeline
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            objective='binary',
            class_weight='balanced',
            verbosity=-1,
            random_state=42,
            **study.best_params
        ))
    ])
    
    print("Fitting Champion Pipeline on entire training set...")
    best_pipeline.fit(X_train, y_train)
    return best_pipeline

def find_optimal_threshold(pipeline, X_val, y_val):
    """Sweeps 100 thresholds to find the peak F1 Score."""
    print("Running Granular 100-Threshold Sweep...")
    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0.1, 0.9, 100)
    
    for thresh in thresholds:
        hard_predictions = (val_probabilities >= thresh).astype(int)
        current_f1 = f1_score(y_val, hard_predictions)
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh

    print("-" * 40)
    print(f"🔥 Peak F1 Score on Test Set: {best_f1:.4f}")
    print(f"🔪 Optimal Business Cutoff Threshold: {best_threshold:.4f}")
    print("-" * 40)
    return best_threshold, best_f1