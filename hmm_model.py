# import numpy as np
# import joblib
# import os
# from hmmlearn.hmm import GaussianHMM
# from typing import Tuple, Optional

# class RegimeHMM:
#     def __init__(self, n_components: int = 2, random_state: int = 42):
#         """
#         Initializes the Hidden Markov Model engine.

#         Args:
#             n_components: The number of hidden regimes to detect (default is 2).
#             random_state: Seed for reproducibility in probability matrices.
#         """
#         self.n_components = n_components

#         # We use a Gaussian HMM because our features (returns/vol) are continuous numbers.
#         # 'full' covariance means returns and volatility are allowed to be correlated.
#         self.model = GaussianHMM(
#             n_components=n_components,
#             covariance_type="full",
#             n_iter=1000,          # High iteration limit to ensure mathematical convergence
#             random_state=random_state
#         )
#         self.is_trained = False

#     def fit(self, features_matrix: np.ndarray) -> bool:
#         """
#         Trains the HMM on a large matrix of historical [log_return, volatility] data.

#         Args:
#             features_matrix: A 2D numpy array of shape (n_samples, n_features).

#         Returns:
#             True if the model converged successfully, False otherwise.
#         """
#         print("[BRAIN INFO] Commencing HMM matrix calibration...")
#         try:
#             self.model.fit(features_matrix)
#             self.is_trained = True
#             print(f"[BRAIN INFO] HMM Calibration complete. Model Converged: {self.model.monitor_.converged}")
#             return self.model.monitor_.converged
#         except Exception as e:
#             print(f"[CRITICAL ERROR] HMM mathematically failed to converge: {str(e)}")
#             return False

#     def predict_current_state(self, recent_features: np.ndarray) -> Tuple[int, np.ndarray]:
#         """
#         Calculates the probability of the current regime based on the latest observation window.

#         Args:
#             recent_features: A 2D numpy array of the most recent [log_return, volatility] data.

#         Returns:
#             A tuple containing: (Predicted State Integer, Array of Probabilities for all states)
#         """
#         if not self.is_trained:
#             raise RuntimeError("CRITICAL: Attempted to predict with an untrained HMM engine.")

#         # predict_proba returns a matrix of probabilities for each timestep.
#         # We only care about the very last timestep [-1].
#         probabilities = self.model.predict_proba(recent_features)[-1]

#         # The predicted state is the one with the highest mathematical probability
#         current_state = int(np.argmax(probabilities))

#         return current_state, probabilities

#     def save_model(self, filepath: str) -> None:
#         """Serializes the trained mathematical model to disk."""
#         if not self.is_trained:
#             print("[WARNING] Attempting to save an untrained model. Aborting.")
#             return

#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         joblib.dump(self.model, filepath)
#         print(f"[SYSTEM INFO] HMM Engine safely stored at: {filepath}")

#     def load_model(self, filepath: str) -> bool:
#         """Loads a pre-trained model from disk into RAM."""
#         if not os.path.exists(filepath):
#             print(f"[WARNING] No pre-trained model found at: {filepath}")
#             return False

#         try:
#             self.model = joblib.load(filepath)
#             self.is_trained = True
#             print("[SYSTEM INFO] Pre-trained HMM Engine successfully loaded into RAM.")
#             return True
#         except Exception as e:
#             print(f"[CRITICAL ERROR] Failed to load HMM file: {str(e)}")
#             return False

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore convergence warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)

class InstitutionalHMMEngine:
    """
    Production-Grade Hidden Markov Model (HMM) Market Regime Detection Engine.
    Features: Dynamic State Selection (AIC/BIC), Feature Standardisation,
    and Posterior Probability Estimation for Institutional Position Sizing.
    """
    def __init__(self, min_components=2, max_components=6, covariance_type='full', n_iter=1000, random_state=42):
        self.min_components = min_components
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        # Scaling & Model Placeholders
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_n_components = None

    def _calculate_parameters_count(self, n_states, n_features):
        """Calculates total trainable parameters to compute exact AIC/BIC metrics."""
        start_prob_params = n_states - 1
        trans_matrix_params = n_states * (n_states - 1)
        means_params = n_states * n_features

        if self.covariance_type == 'full':
            cov_params = n_states * (n_features * (n_features + 1) // 2)
        elif self.covariance_type == 'diag':
            cov_params = n_states * n_features
        else:
            cov_params = n_states * n_features # Fallback approximation

        total_params = start_prob_params + trans_matrix_params + means_params + cov_params
        return total_params

    def fit(self, df, feature_cols):
        """
        Standardises features, iterates through components (min to max),
        and selects the mathematical best fit using the Bayesian Information Criterion (BIC).
        """
        # 1. Clean data and extract features
        data_clean = df[feature_cols].dropna().copy()
        X = data_clean.values
        T = X.shape[0]
        n_features = X.shape[1]

        # 2. Institutional Scaling Layer (Prevents feature dominance)
        X_scaled = self.scaler.fit_transform(X)

        best_bic = float('inf')
        best_model = None
        results = []

        # 3. Dynamic Optimization Loop (Grid Search for Best Regimes)
        for n in range(self.min_components, self.max_components + 1):
            try:
                model = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state
                )
                model.fit(X_scaled)

                log_likelihood = model.score(X_scaled)
                p = self._calculate_parameters_count(n, n_features)

                # Metric Formulas
                aic = -2 * log_likelihood + 2 * p
                bic = -2 * log_likelihood + p * np.log(T)

                results.append({'states': n, 'BIC': bic, 'AIC': aic, 'LL': log_likelihood})

                # We prioritise BIC to heavily penalise over-fitting in noisy financial data
                if bic < best_bic:
                    best_bic = bic
                    self.best_model = model
                    self.best_n_components = n
            except Exception as e:
                continue

        print("\n======================= HMM REGIME CRITERIA SELECTION =======================")
        print(pd.DataFrame(results).to_string(index=False))
        print(f"\n[SELECTED OPTIMAL ENGINE]: {self.best_n_components} Hidden Market Regimes")
        print("=============================================================================")
        return self

    def predict_regimes(self, df, feature_cols):
        """
        Generates scaled predictions and returns states along with confidence metrics.
        """
        if self.best_model is None:
            raise ValueError("Engine must be fitted before predicting regimes.")

        # Clean data and keep original indices
        valid_idx = df[feature_cols].dropna().index
        X = df.loc[valid_idx, feature_cols].values

        # Scale input data using fitted scaler
        X_scaled = self.scaler.transform(X)

        # Predict hidden sequences & calculate posterior probabilities
        hidden_states = self.best_model.predict(X_scaled)
        posterior_probs = self.best_model.predict_proba(X_scaled)

        # Construct institutional payload
        output_df = pd.DataFrame(index=valid_idx)
        output_df['market_regime'] = hidden_states

        # Append state confidence vectors
        for state_idx in range(self.best_n_components):
            output_df[f'regime_prob_{state_idx}'] = posterior_probs[:, state_idx]

        # Compute maximum assignment confidence
        output_df['regime_confidence'] = output_df[[f'regime_prob_{i}' for i in range(self.best_n_components)]].max(axis=1)

        return output_df