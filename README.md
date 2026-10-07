# FinGuard Pulse

**Cost-aware payment fraud decisioning with explainable alerts and drift monitoring**

This project is a prototype AI system that scores digital payments for fraud risk, decides which payments to flag by comparing expected rupees saved with the expected cost of friction, explains what drives the risk, and monitors for fraud tactic drift. It moves beyond standard accuracy metrics (like AUC) to focus on the actual business value (net rupees saved).

## Problem Statement

Payment providers typically judge fraud models on accuracy or AUC. However, fraud is rare, and the two types of mistakes have very different costs: a missed fraud costs the whole transaction amount, while a false alarm costs customer friction and handling time. This project asks: **for which payments does holding the transaction save more money than it costs?**

## Solution Architecture

The FinGuard Pulse system consists of four main parts:

1.  **Risk Engine**: A gradient-boosted classifier (`HistGradientBoostingClassifier`) trained on labeled payment history to output the probability of fraud. A logistic regression model is also included as a baseline.
2.  **Expected-Value Decision Rule**: Instead of a flat probability cut-off (e.g., 0.5), a payment is flagged when the expected loss prevented is larger than the expected cost of flagging: `p x amount > (1 - p) x friction + ops_cost`. This means large transactions are flagged at a much lower risk score than small ones.
3.  **Explainability**: Permutation importance is used to show which signals (e.g., new device, amount, hour) drive the model's decisions, helping analysts understand why a transaction was flagged.
4.  **Drift Monitor**: Calculates the Population Stability Index (PSI) per feature against the training data. Shifts in fraud tactics (e.g., a sudden rise in "collect request" scams) trigger drift alerts, indicating a need for retraining.

## Project Structure

*   `data_generator.py`: Generates synthetic payment data for two periods, simulating "account takeover" fraud in Period 1 and a drift to "collect-request" scams in Period 2.
*   `finguard.py`: Contains the core machine learning logic, including model training, the expected-value decision rule, policy evaluation, and drift calculation.
*   `app.py`: A Streamlit dashboard with three interactive tabs:
    *   **Policy Simulator**: Compares the net benefit of various decision policies.
    *   **Transaction Scorer**: Allows manual input of transaction details to score it and see the explainability.
    *   **Drift Monitor**: Visualizes PSI across features to detect data drift.
*   `requirements.txt`: List of Python dependencies.

## Setup and Installation

1.  **Clone the repository** (if pushed to version control):
    ```bash
    # git clone <your-repo-url>
    cd FinGuardPulse
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    
    # On Windows:
    .\venv\Scripts\activate
    
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the synthetic data**:
    ```bash
    python data_generator.py
    ```
    *(This will create `period1_data.csv` and `period2_data.csv` in the project root)*

5.  **Run the Streamlit Dashboard**:
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser (usually at `http://localhost:8501`).

## Technology Stack

*   **Language**: Python
*   **Data Manipulation**: pandas, NumPy
*   **Machine Learning**: scikit-learn (`HistGradientBoostingClassifier`, `LogisticRegression`)
*   **Dashboard**: Streamlit
*   **Visualization**: Matplotlib
