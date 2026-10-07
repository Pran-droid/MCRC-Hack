import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from finguard import (
    train_models, 
    apply_expected_value_rule, 
    evaluate_policy, 
    calculate_drift, 
    get_permutation_importance
)

st.set_page_config(page_title="FinGuard Pulse", layout="wide")

st.title("FinGuard Pulse")
st.subheader("Cost-aware payment fraud decisioning with explainable alerts and drift monitoring")

# --- DATA LOADING & MODEL TRAINING ---
@st.cache_data
def load_and_train():
    features = ['amount', 'amount_ratio', 'hour', 'new_device', 'new_payee', 
                'collect_request', 'txns_last_hour', 'geo_distance_km', 
                'account_age_days', 'failed_pin_24h']
    
    # Load Data
    df_p1 = pd.read_csv("period1_data.csv")
    df_p2 = pd.read_csv("period2_data.csv")
    
    # Train/Val/Test split on Period 1
    # Train: 120k, Val: 40k, Test: 40k
    train_df = df_p1.iloc[:120000]
    val_df = df_p1.iloc[120000:160000]
    test_df = df_p1.iloc[160000:]
    
    gb_model, lr_model = train_models(train_df, features, 'is_fraud')
    
    return train_df, test_df, df_p2, gb_model, lr_model, features

with st.spinner("Loading data and training models..."):
    train_df, test_df, df_p2, gb_model, lr_model, features = load_and_train()

tab1, tab2, tab3 = st.tabs(["Policy Simulator", "Transaction Scorer", "Drift Monitor"])

# --- TAB 1: POLICY SIMULATOR ---
with tab1:
    st.header("Policy Simulator")
    st.markdown("Compare different decision policies on a held-out test set (40,000 payments) from Period 1.")
    
    friction_cost = st.sidebar.number_input("Friction Cost (INR)", value=80, step=10)
    ops_cost = st.sidebar.number_input("Ops Handling Cost (INR)", value=10, step=5)
    
    X_test = test_df[features]
    y_test = test_df['is_fraud']
    amounts = test_df['amount'].values
    
    # Get probabilities
    probs_gb = gb_model.predict_proba(X_test)[:, 1]
    probs_lr = lr_model.predict_proba(X_test)[:, 1]
    
    results = []
    
    # 1. Approve everything
    mask_approve_all = np.zeros(len(y_test), dtype=bool)
    results.append({
        "Policy": "Approve everything (no screening)",
        **evaluate_policy(mask_approve_all, y_test, amounts, friction_cost, ops_cost)
    })
    
    # 2. Hand-written rules (e.g. amount > 2000 & new_payee == 1)
    mask_hand_written = (X_test['amount'] > 2000) & (X_test['new_payee'] == 1)
    results.append({
        "Policy": "Hand-written rules",
        **evaluate_policy(mask_hand_written, y_test, amounts, friction_cost, ops_cost)
    })
    
    # 3. GB default 0.5 cut-off
    mask_gb_05 = probs_gb > 0.5
    results.append({
        "Policy": "Gradient boosting, default 0.5 cut-off",
        **evaluate_policy(mask_gb_05, y_test, amounts, friction_cost, ops_cost)
    })
    
    # 4. GB tuned flat cut-off (0.059)
    mask_gb_tuned = probs_gb > 0.059
    results.append({
        "Policy": "Gradient boosting, tuned flat cut-off (0.059)",
        **evaluate_policy(mask_gb_tuned, y_test, amounts, friction_cost, ops_cost)
    })
    
    # 5. LR + expected-value rule
    mask_lr_ev = apply_expected_value_rule(probs_lr, amounts, friction_cost, ops_cost)
    results.append({
        "Policy": "Logistic regression + expected-value rule",
        **evaluate_policy(mask_lr_ev, y_test, amounts, friction_cost, ops_cost)
    })
    
    # 6. GB + expected-value rule
    mask_gb_ev = apply_expected_value_rule(probs_gb, amounts, friction_cost, ops_cost)
    results.append({
        "Policy": "Gradient boosting + expected-value rule",
        **evaluate_policy(mask_gb_ev, y_test, amounts, friction_cost, ops_cost)
    })
    
    results_df = pd.DataFrame(results)
    
    # Formatting for display
    display_df = results_df.copy()
    display_df['Net benefit per 100,000 (INR)'] = display_df['net_benefit_scaled'].apply(lambda x: f"{int(x):,}")
    display_df['Fraud caught'] = display_df['fraud_caught_pct'].apply(lambda x: f"{x:.1f}%")
    display_df['Fraud value caught'] = display_df['fraud_value_caught_pct'].apply(lambda x: f"{x:.1f}%")
    display_df['False alarms per 100,000'] = display_df['false_alarms_scaled'].apply(lambda x: f"{int(x):,}")
    
    st.table(display_df[['Policy', 'Net benefit per 100,000 (INR)', 'Fraud caught', 'Fraud value caught', 'False alarms per 100,000']])
    
    # Plotting Net Benefit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(results_df['Policy'], results_df['net_benefit_scaled'], color='teal')
    ax.set_xlabel('Net Benefit per 100,000 (INR)')
    ax.set_title('Net Benefit Comparison across Policies')
    plt.tight_layout()
    st.pyplot(fig)


# --- TAB 2: TRANSACTION SCORER ---
with tab2:
    st.header("Single Transaction Scorer")
    st.markdown("Input transaction details to score it using the Gradient Boosting model and the expected-value decision rule.")
    
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Amount (INR)", min_value=1, value=5000)
        amt_ratio = st.number_input("Amount vs Usual (Ratio)", min_value=0.0, value=2.5)
        hour = st.slider("Hour of Day", 0, 23, 2)
        new_device = st.selectbox("New Device", [0, 1], index=1)
        new_payee = st.selectbox("New Payee", [0, 1], index=1)
        
    with col2:
        col_req = st.selectbox("Collect Request", [0, 1], index=0)
        txns_lh = st.number_input("Transactions in Last Hour", min_value=0, value=0)
        geo_dist = st.number_input("Distance from Usual Location (km)", min_value=0.0, value=15.0)
        acc_age = st.number_input("Account Age (Days)", min_value=1, value=365)
        fail_pin = st.number_input("Failed PIN in 24h", min_value=0, value=0)
        
    if st.button("Score Transaction"):
        input_data = pd.DataFrame([[amt, amt_ratio, hour, new_device, new_payee, col_req, txns_lh, geo_dist, acc_age, fail_pin]], columns=features)
        
        prob = gb_model.predict_proba(input_data)[0, 1]
        decision = apply_expected_value_rule(np.array([prob]), np.array([amt]), friction_cost, ops_cost)[0]
        
        st.subheader("Results")
        st.write(f"**Fraud Probability:** {prob:.4f}")
        st.write(f"**Expected Value Rule Decision:** {'**FLAG (Block)**' if decision else '**APPROVE**'}")
        
        if decision:
            st.error("Transaction Flagged for Fraud Review.")
            savings = prob * amt
            cost = (1 - prob) * friction_cost + ops_cost
            st.info(f"Expected Savings: {savings:.2f} INR | Expected Cost: {cost:.2f} INR")
        else:
            st.success("Transaction Approved.")
            
        st.subheader("Global Model Explainability (Permutation Importance)")
        with st.spinner("Calculating permutation importance..."):
            importance_df = get_permutation_importance(gb_model, train_df[features].iloc[:5000], train_df['is_fraud'].iloc[:5000])
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.barh(importance_df['feature'], importance_df['importance'], color='coral')
            ax2.set_xlabel('Mean Accuracy Decrease')
            ax2.invert_yaxis()
            st.pyplot(fig2)


# --- TAB 3: DRIFT MONITOR ---
with tab3:
    st.header("Drift Monitor")
    st.markdown("Monitoring data distribution changes (drift) between Period 1 and Period 2.")
    
    with st.spinner("Calculating PSI..."):
        drift_results = calculate_drift(test_df, df_p2, features)
        
        drift_df = pd.DataFrame(list(drift_results.items()), columns=['Feature', 'PSI'])
        drift_df = drift_df.sort_values('PSI', ascending=False)
        
        st.write("Population Stability Index (PSI) > 0.25 indicates significant drift.")
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        
        # Color code bars
        colors = ['red' if psi > 0.25 else 'orange' if psi > 0.1 else 'grey' for psi in drift_df['PSI']]
        
        ax3.bar(drift_df['Feature'], drift_df['PSI'], color=colors)
        ax3.axhline(y=0.25, color='r', linestyle='--', label='Major Shift (0.25)')
        ax3.axhline(y=0.10, color='orange', linestyle='--', label='Minor Shift (0.10)')
        ax3.set_ylabel('PSI')
        ax3.set_xticklabels(drift_df['Feature'], rotation=45, ha='right')
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.write("**Drift Alert:** In Period 2, `collect_request` payments rise significantly, reflecting a shift to social engineering scams.")
