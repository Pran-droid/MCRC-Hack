import pandas as pd
import numpy as np

def generate_data(n_samples, fraud_rate, period):
    np.random.seed(42 if period == 1 else 43)
    
    # Base distributions for non-fraud
    df = pd.DataFrame({
        'amount': np.random.lognormal(mean=5, sigma=1, size=n_samples), # Typical amounts 150-1000
        'amount_ratio': np.random.lognormal(mean=0, sigma=0.5, size=n_samples), # Ratio to usual amount ~1.0
        'hour': np.random.randint(0, 24, size=n_samples),
        'new_device': np.random.binomial(1, 0.05, size=n_samples), # 5% of normal txns are on new devices
        'new_payee': np.random.binomial(1, 0.10, size=n_samples), # 10% to new payees
        'collect_request': np.random.binomial(1, 0.08 if period == 1 else 0.30, size=n_samples), # 8% P1, 30% P2
        'txns_last_hour': np.random.poisson(0.5, size=n_samples),
        'geo_distance_km': np.random.lognormal(mean=1, sigma=1, size=n_samples),
        'account_age_days': np.random.randint(1, 3650, size=n_samples),
        'failed_pin_24h': np.random.poisson(0.1, size=n_samples)
    })
    
    # Assign fraud labels
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    df['is_fraud'] = 0
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    # Inject fraud patterns
    if period == 1:
        # Pattern: Account Takeover (large amounts, new device, new payee, night hours)
        df.loc[fraud_indices, 'amount'] = np.random.lognormal(mean=9, sigma=1, size=n_fraud) # ~8000
        df.loc[fraud_indices, 'amount_ratio'] = np.random.lognormal(mean=2, sigma=0.5, size=n_fraud)
        df.loc[fraud_indices, 'new_device'] = np.random.binomial(1, 0.8, size=n_fraud)
        df.loc[fraud_indices, 'new_payee'] = np.random.binomial(1, 0.9, size=n_fraud)
        df.loc[fraud_indices, 'hour'] = np.random.choice([22, 23, 0, 1, 2, 3, 4], size=n_fraud)
    else:
        # Pattern: Collect-request scams (smaller amounts, daytime, own device, collect_request)
        df.loc[fraud_indices, 'amount'] = np.random.lognormal(mean=7, sigma=1, size=n_fraud) # ~1100
        df.loc[fraud_indices, 'amount_ratio'] = np.random.lognormal(mean=1, sigma=0.5, size=n_fraud)
        df.loc[fraud_indices, 'new_device'] = np.random.binomial(1, 0.1, size=n_fraud) # Mostly own device
        df.loc[fraud_indices, 'collect_request'] = np.random.binomial(1, 0.95, size=n_fraud) # Almost always collect request
        df.loc[fraud_indices, 'hour'] = np.random.randint(9, 18, size=n_fraud) # Daytime
        
    return df

if __name__ == "__main__":
    print("Generating Period 1 data...")
    df_p1 = generate_data(200000, 0.01, period=1)
    df_p1.to_csv("period1_data.csv", index=False)
    
    print("Generating Period 2 data...")
    df_p2 = generate_data(100000, 0.0123, period=2)
    df_p2.to_csv("period2_data.csv", index=False)
    
    print("Done!")
