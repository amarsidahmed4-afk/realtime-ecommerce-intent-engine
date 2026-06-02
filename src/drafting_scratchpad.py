import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline



# change preprocessor : handle different cardinalities for categorical features

# 1. Be ruthlessly explicit to prevent leaks and overlaps
num_cols = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]
low_cardinality_cols = ['Weekend', 'VisitorType']
high_cardinality_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Month']

# 2. Rebuild the Pipelines exactly as before
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_low_pipeline = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

cat_high_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('target_encoder', TargetEncoder(target_type='binary', smooth='auto')),
    ('scaler', StandardScaler()) # Essential for TargetEncoder probabilities
])

# 3. Assemble with explicit lists, completely ditching make_column_selector
advanced_preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat_low', cat_low_pipeline, low_cardinality_cols),
        ('cat_high', cat_high_pipeline, high_cardinality_cols)
    ],
    remainder='drop' 
)

# 4. The Final Master Pipeline
advanced_pipeline = Pipeline(steps=[
    ('preprocessor', advanced_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])


###########################################################################################################

# loading Online Shoppers Intention DataSet
df=pd.read_csv("online_shoppers_intention.csv")

# Filter out all rows where the Month is 'Feb'
df = df[df['Month'] != 'Feb'].copy()

# Reset the index (Best practice after dropping rows to prevent index alignment bugs later)
df.reset_index(drop=True, inplace=True)

# Separate features (X) and target (y)
X = df.drop('Revenue', axis=1)
y = df['Revenue'].astype(int) # Ensure the target is 0 or 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting the model
advanced_pipeline.fit(X_train, y_train)

# cross validation
cv_results = cross_validate(advanced_pipeline, 
X, y, # Note: using the full X and y here for CV
        cv=10, 
        return_estimator=True,
        n_jobs=-1 # Uses all CPU cores for speed
    )

# 3. Calculate mean cross-validation accuracy
mean_accuracy = cv_results['test_score'].mean()
mean_accuracy
cv_results['test_score']
    
# Check the accuracy
score_advanced = advanced_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {score_advanced:.3f}")

# predict
y_pred = advanced_pipeline.predict(X_test)

# confusion matrix
print(confusion_matrix(y_test, y_pred))
# Look at the F1-Score, Precision, and Recall for class 1 (Conversions)
print(classification_report(y_test, y_pred))


# threshold recall/precision
# 1. Extract the raw probabilities for Class 1 (Conversion)
# predict_proba returns an array of shape (n_samples, 2): [prob_class_0, prob_class_1]
# We slice [:, 1] to grab only the conversion probabilities
y_probs = advanced_pipeline.predict_proba(X_test)[:, 1]

# 2. Define a range of thresholds to test (e.g., from 0.30 to 0.80)
thresholds = np.arange(0.30, 0.85, 0.05)
results = []

# 3. Loop through each threshold and calculate the new metrics
for thresh in thresholds:
    # If probability >= threshold, predict 1. Otherwise, predict 0.
    y_pred_custom = (y_probs >= thresh).astype(int)
    
    # Calculate metrics (zero_division=0 prevents warnings if a threshold is too high)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    
    results.append({
        'Threshold': round(thresh, 2),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-Score': round(f1, 3)
    })

# 4. Display as a clean DataFrame
threshold_df = pd.DataFrame(results)
display(threshold_df)

# visual
# Find the exact row where the F1-Score is at its maximum
best_idx = threshold_df['F1-Score'].idxmax()
best_thresh = threshold_df.loc[best_idx, 'Threshold']
best_f1 = threshold_df.loc[best_idx, 'F1-Score']

# Set up the plot style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plot the three metrics
plt.plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision', 
         color='#3498db', linewidth=2.5, marker='o')
plt.plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall', 
         color='#e74c3c', linewidth=2.5, marker='s')
plt.plot(threshold_df['Threshold'], threshold_df['F1-Score'], label='F1-Score', 
         color='#2ecc71', linewidth=2.5, linestyle='--', marker='^')

# Highlight the optimal F1-Score threshold
plt.axvline(x=best_thresh, color='black', linestyle=':', linewidth=2, 
            label=f'Max F1 Peak (Thresh: {best_thresh:.2f})')
plt.scatter(best_thresh, best_f1, color='black', s=100, zorder=5)

# Formatting
plt.title('Precision-Recall Trade-off across Decision Thresholds', fontsize=16, pad=15)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(threshold_df['Threshold'])
plt.ylim(0.0, 1.05)
plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()

# Show the plot
plt.show()

print(f"The F1-Score peaks at a threshold of {best_thresh:.2f} with a score of {best_f1:.3f}")





# switching to LightGBM

# 2. Assemble the LightGBM Pipeline
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', advanced_preprocessor), # Re-using our exact same preprocessor!
    # class_weight='balanced' works identically here to fight class imbalance
    ('classifier', LGBMClassifier(class_weight='balanced', random_state=42, n_estimators=200))
])

# 3. Fit the beast
lgbm_pipeline.fit(X_train, y_train)

# 4. Extract raw probabilities
lgbm_probs = lgbm_pipeline.predict_proba(X_test)[:, 1]

# 5. Run the threshold loop to map the new Precision/Recall trade-off
lgbm_results = []
for thresh in np.arange(0.30, 0.85, 0.05):
    y_pred_custom = (lgbm_probs >= thresh).astype(int)
    
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    
    lgbm_results.append({
        'Threshold': round(thresh, 2),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-Score': round(f1, 3)
    })

# 6. Display the new map
lgbm_threshold_df = pd.DataFrame(lgbm_results)
display(lgbm_threshold_df)


# visual
# Find the exact row where the F1-Score is at its maximum
best_idx = lgbm_threshold_df['F1-Score'].idxmax()
best_thresh = lgbm_threshold_df.loc[best_idx, 'Threshold']
best_f1 = lgbm_threshold_df.loc[best_idx, 'F1-Score']

# Set up the plot style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plot the three metrics
plt.plot(lgbm_threshold_df['Threshold'], lgbm_threshold_df['Precision'], label='Precision', 
         color='#3498db', linewidth=2.5, marker='o')
plt.plot(lgbm_threshold_df['Threshold'], lgbm_threshold_df['Recall'], label='Recall', 
         color='#e74c3c', linewidth=2.5, marker='s')
plt.plot(lgbm_threshold_df['Threshold'], lgbm_threshold_df['F1-Score'], label='F1-Score', 
         color='#2ecc71', linewidth=2.5, linestyle='--', marker='^')

# Highlight the optimal F1-Score threshold
plt.axvline(x=best_thresh, color='black', linestyle=':', linewidth=2, 
            label=f'Max F1 Peak (Thresh: {best_thresh:.2f})')
plt.scatter(best_thresh, best_f1, color='black', s=100, zorder=5)

# Formatting
plt.title('Precision-Recall Trade-off across Decision Thresholds', fontsize=16, pad=15)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(lgbm_threshold_df['Threshold'])
plt.ylim(0.0, 1.05)
plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()

# Show the plot
plt.show()

print(f"The F1-Score peaks at a threshold of {best_thresh:.2f} with a score of {best_f1:.3f}")







# tuning LightGBM

# 1. Define the Optuna Objective Function
def objective(trial):
    # Tell Optuna which hyperparameters to tweak and the ranges to guess within
    params = {
        'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 100, 500),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.2, log=True),
        'classifier__max_depth': trial.suggest_int('classifier__max_depth', 3, 12),
        'classifier__num_leaves': trial.suggest_int('classifier__num_leaves', 20, 150),
        'classifier__min_child_samples': trial.suggest_int('classifier__min_child_samples', 10, 100)
    }
    
    # 2. Rebuild the pipeline with the current trial's parameters
    model = Pipeline(steps=[
        ('preprocessor', advanced_preprocessor), # Re-using our trusted preprocessor
        ('classifier', LGBMClassifier(class_weight='balanced', random_state=42))
    ])
    
    # Inject Optuna's guesses into the pipeline
    model.set_params(**params)
    
    # 3. Cross-validate and return the F1-Score for Optuna to evaluate
    # Using cv=3 keeps the tuning fast; scoring='f1' tells it to focus on Class 1
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)
    
    return score.mean()

# 4. Create the study and turn it on! 
# (n_trials=20 is a great starting point to see it in action without waiting an hour)
print("Starting Optuna optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("\n--- Optuna Tuning Complete ---")
print(f"Best F1-Score found: {study.best_value:.3f}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")




# 1. Update the pipeline with Optuna's exact findings
optimized_lgbm = Pipeline(steps=[
    ('preprocessor', advanced_preprocessor), # Re-using our trusted preprocessor
    ('classifier', LGBMClassifier(
        class_weight='balanced',
        n_estimators=273,
        learning_rate=0.047897,
        max_depth=11,
        num_leaves=49,
        min_child_samples=12,
        random_state=42
    ))
])

# 2. Fit the tuned beast
optimized_lgbm.fit(X_train, y_train)

# 3. Extract raw probabilities
opt_probs = optimized_lgbm.predict_proba(X_test)[:, 1]

# 4. Run the threshold loop
opt_results = []
thresholds = np.arange(0.30, 0.85, 0.05)

for thresh in thresholds:
    y_pred_custom = (opt_probs >= thresh).astype(int)
    
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    
    opt_results.append({
        'Threshold': round(thresh, 2),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-Score': round(f1, 3)
    })

opt_df = pd.DataFrame(opt_results)

# 5. Visualize the new optimal curve
best_idx = opt_df['F1-Score'].idxmax()
best_thresh = opt_df.loc[best_idx, 'Threshold']
best_f1 = opt_df.loc[best_idx, 'F1-Score']
best_prec = opt_df.loc[best_idx, 'Precision']
best_rec = opt_df.loc[best_idx, 'Recall']

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

plt.plot(opt_df['Threshold'], opt_df['Precision'], label='Precision', color='#3498db', linewidth=2.5, marker='o')
plt.plot(opt_df['Threshold'], opt_df['Recall'], label='Recall', color='#e74c3c', linewidth=2.5, marker='s')
plt.plot(opt_df['Threshold'], opt_df['F1-Score'], label='F1-Score', color='#2ecc71', linewidth=2.5, linestyle='--', marker='^')

plt.axvline(x=best_thresh, color='black', linestyle=':', linewidth=2, 
            label=f'Max F1 Peak (Thresh: {best_thresh:.2f})')
plt.scatter(best_thresh, best_f1, color='black', s=100, zorder=5)

plt.title('Optimized LightGBM: Precision-Recall Trade-off', fontsize=16, pad=15)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(thresholds)
plt.ylim(0.0, 1.05)
plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.show()

print(f"NEW OPTIMAL SWEET SPOT:")
print(f"Threshold: {best_thresh:.2f}")
print(f"F1-Score:  {best_f1:.3f}")
print(f"Precision: {best_prec:.3f}")
print(f"Recall:    {best_rec:.3f}")
