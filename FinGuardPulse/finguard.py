import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

def train_models(df, features, target):
    X = df[features]
    y = df[target]
    
    # Gradient Boosting Model
    gb_model = HistGradientBoostingClassifier(
        max_depth=4, 
        max_iter=200, 
        learning_rate=0.06, 
        l2_regularization=1.0, 
        random_state=42
    )
    gb_model.fit(X, y)
    
    # Logistic Regression Model (with scaling)
    lr_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42))
    ])
    lr_model.fit(X, y)
    
    return gb_model, lr_model

def apply_expected_value_rule(probs, amounts, friction_cost=80, ops_cost=10):
    # p * amount > (1 - p) * friction + ops_cost
    # Using element-wise numpy operations
    expected_savings = probs * amounts
    expected_costs = (1 - probs) * friction_cost + ops_cost
    return expected_savings > expected_costs

def evaluate_policy(flagged_mask, is_fraud, amounts, friction_cost=80, ops_cost=10, scale_to=100000):
    n_payments = len(is_fraud)
    
    # Identify True Positives, False Positives
    true_positives = flagged_mask & is_fraud
    false_positives = flagged_mask & (~is_fraud)
    
    # Calculate costs and savings
    amount_saved = amounts[true_positives].sum()
    handling_costs = flagged_mask.sum() * ops_cost
    friction_costs = false_positives.sum() * friction_cost
    
    net_benefit = amount_saved - handling_costs - friction_costs
    
    # Metrics scaling
    scaling_factor = scale_to / n_payments
    
    fraud_cases_caught = true_positives.sum()
    total_fraud_cases = is_fraud.sum()
    fraud_caught_rate = fraud_cases_caught / total_fraud_cases if total_fraud_cases > 0 else 0
    
    fraud_value_caught = amount_saved
    total_fraud_value = amounts[is_fraud].sum()
    fraud_value_rate = fraud_value_caught / total_fraud_value if total_fraud_value > 0 else 0
    
    false_alarms_scaled = false_positives.sum() * scaling_factor
    
    return {
        'net_benefit_scaled': net_benefit * scaling_factor,
        'fraud_caught_pct': fraud_caught_rate * 100,
        'fraud_value_caught_pct': fraud_value_rate * 100,
        'false_alarms_scaled': false_alarms_scaled
    }

def calculate_psi(expected, actual, buckets=10):
    # Determine bin edges based on the expected data (period 1)
    # Using quantiles for numerical, unique values for binary
    if len(np.unique(expected)) <= 2: # Binary
        bins = np.array([-0.1, 0.5, 1.1])
    else:
        bins = np.quantile(expected, np.linspace(0, 1, buckets + 1))
        # Ensure bins are unique to avoid pandas cut errors
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0 # No variance
        # Small padding on ends
        bins[0] -= 0.001
        bins[-1] += 0.001

    # Calculate distributions
    expected_percents = np.histogram(expected, bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins)[0] / len(actual)
    
    # Avoid zero division
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)

def calculate_drift(df_p1, df_p2, features):
    drift_results = {}
    for feature in features:
        psi = calculate_psi(df_p1[feature], df_p2[feature])
        drift_results[feature] = psi
    return drift_results

def get_permutation_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    return pd.DataFrame({
        'feature': X.columns,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
