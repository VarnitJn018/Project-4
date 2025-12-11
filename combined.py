# -*- coding: utf-8 -*-
"""
Merged Optimization Suite
Combines:
1. GPyTorch ExactGP Training & Prediction
2. BoTorch Optimization with Custom Constraints & LLM Assistant
"""

import sys
import os
import warnings
import json
import re
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Tuple, Any

# Machine Learning Imports
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# BoTorch Imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- CONFIGURATION ---
OPENAI_API_KEY = "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"

warnings.filterwarnings('ignore')

# --- DEPENDENCY CHECKS ---
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ 'openai' library not found. LLM features will be disabled. Run 'pip install openai'")

try:
    import tiktoken
except ImportError:
    tiktoken = None


# =============================================================================
# PART 1: SHARED GP MODEL ARCHITECTURE (From Scripts 1 & 2)
# =============================================================================

class ExactGPModel(gpytorch.models.ExactGP):
    """
    Standard Exact GP Model used for training on static datasets 
    and predicting on Excel files.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# =============================================================================
# PART 2: TRAINING AND PREDICTION UTILITIES (From Scripts 1 & 2)
# =============================================================================

def train_gp_on_file(csv_path="datafull.csv", model_save_path="gp_model.pth", scaler_save_path="x_scaler.pkl"):
    """
    Trains a GP model on the first 650 rows of a CSV and tests on the rest.
    Saves the model and scaler.
    """
    print(f"\n--- Starting GP Training on {csv_path} ---")
    try:
        df = pd.read_csv(csv_path).dropna(how="all")
    except FileNotFoundError:
        print(f"❌ File {csv_path} not found. Skipping training.")
        return

    # Split Data (Hardcoded based on Script 2 logic)
    # Training set = rows 0..649
    if len(df) > 650:
        X_train_np = df.iloc[:650, 0:10].values.astype(np.float64)
        y_train_np = df.iloc[:650, 10].values.astype(np.float64)
        
        # Test set = rows 650..811
        X_test_np = df.iloc[650:812, 0:10].values.astype(np.float64)
        y_test_np = df.iloc[650:812, 10].values.astype(np.float64)
    else:
        # Fallback for smaller datasets
        split_idx = int(len(df) * 0.8)
        X_train_np = df.iloc[:split_idx, 0:10].values.astype(np.float64)
        y_train_np = df.iloc[:split_idx, 10].values.astype(np.float64)
        X_test_np = df.iloc[split_idx:, 0:10].values.astype(np.float64)
        y_test_np = df.iloc[split_idx:, 10].values.astype(np.float64)

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    joblib.dump(scaler, scaler_save_path)

    # Convert to Torch
    train_X = torch.tensor(X_train_scaled, dtype=torch.double)
    train_Y = torch.tensor(y_train_np, dtype=torch.double)
    
    # Initialize Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_X, train_Y, likelihood).double()

    # Training Loop
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    print("Training GP...")
    for i in range(300):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"Iter {i+1}/300 - Loss {loss.item():.5f}")

    # Save Model
    torch.save({
        "model_state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluation
    model.eval()
    likelihood.eval()
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.double)
    
    with torch.no_grad():
        posterior = model(X_test_t)
        y_pred = posterior.mean.numpy()

    # Metrics & Plotting
    if len(y_test_np) > 1:
        r2 = r2_score(y_test_np, y_pred)
        print(f"R² score on test set: {r2:.4f}")

        plt.figure(figsize=(7, 6))
        plt.scatter(y_test_np, y_pred, alpha=0.6)
        plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'k--', linewidth=2)
        plt.xlabel("Actual Output")
        plt.ylabel("Predicted Output")
        plt.title("GP Predictions vs Actual Values")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def predict_from_excel_file(excel_path, model_path="gp_model.pth", scaler_path="x_scaler.pkl", output_file="new_outputs.xlsx"):
    """
    Loads trained GP model + scaler and predicts outputs for an Excel file.
    """
    print(f"\n--- Predicting on {excel_path} ---")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Model or scaler not found. Please train the model first.")
        return None

    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"❌ Input file {excel_path} not found.")
        return None

    # Determine input columns (Assuming columns 1 to 11 based on Script 1)
    # Check if shape matches expected dimensions
    try:
        X_new = df.iloc[:, 1:11].values.astype(np.float64)
    except IndexError:
        print("❌ Error: Excel file does not have enough columns. Expecting data in columns 2-11.")
        return None

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_new)
    X_t = torch.tensor(X_scaled, dtype=torch.double)

    # Rebuild model structure
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    dummy_y = torch.zeros(X_t.shape[0], dtype=torch.double)
    model = ExactGPModel(X_t, dummy_y, likelihood).double()

    # Load State
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        posterior = model(X_t)
        y_pred = posterior.mean.numpy()

    df["Predicted_Output_GP"] = y_pred
    df.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    return df


# =============================================================================
# PART 3: BOTORCH OPTIMIZATION & LLM ASSISTANT (From Script 3)
# =============================================================================

# --- Prompts ---
PROMPTS = {
    "experiment_overview": """
You are an expert scientist assisting with a chemical optimization experiment.
Goal: Maximize [target].
Description: [description]
Parameters: [parameters_and_bounds]
Constraint: [constraint]
Domain: [domain]
Please provide a brief overview of the experiment.
""",
    "starter": """
Based on the experiment description, generate [n_hypotheses] initial hypotheses for points that might maximize the target.
Consider the chemical properties. Return JSON: {"comment": string, "hypotheses": [{"name", "rationale", "points"}]}.
""",
    "comment_selection": """
Current Iteration: [iteration]
Top qUCB Suggestions:
[suggestions]

Historical Data (Top/Recent):
[dataset]

Analyze these suggestions. Which ones seem most chemically promising?
Return JSON with "comment" and "hypotheses" (where "points" are the selected/refined candidates).
"""
}

# --- Mock Classes for LLM Context ---
class MockParameter:
    def __init__(self, name, bounds, step=None):
        self.name = name
        self.bounds = bounds
        self.step = step
    def get_bounds(self): return self.bounds

class MockConstraint:
    def __init__(self, description): self.description = description

class MockTarget:
    def __init__(self, name): self.name = name

class MockExperiment:
    def __init__(self, name, parameters, target_name):
        self.name = name
        self.parameters = parameters
        self.constraint = MockConstraint("Sum of components (excluding P10) <= 5")
        self.domain = "Chemical Formulation"
        self.target = MockTarget(target_name)
        self.description = "Optimize a chemical formula to maximize yield."

# --- LLM Assistant Class ---
class Assistant:
    def __init__(self, api_key, experiment, log_path, llm_model="gpt-4o-mini"):
        self._api_key = api_key
        self._experiment = experiment
        self._client = OpenAI(api_key=api_key) if HAS_OPENAI else None
        self._model = llm_model
        self._log_path = log_path
        self._chat_history = [{"role": "system", "content": "You are BORA, an AI scientist for Bayesian Optimization."}]

    def _fill_prompt(self, template, extra_subs={}):
        txt = template
        txt = txt.replace("[target]", self._experiment.target.name)
        txt = txt.replace("[description]", self._experiment.description)
        txt = txt.replace("[constraint]", self._experiment.constraint.description)
        txt = txt.replace("[domain]", self._experiment.domain)
        p_str = "".join([f"- {p.name}: bounds={p.get_bounds()}\n" for p in self._experiment.parameters])
        txt = txt.replace("[parameters_and_bounds]", p_str)
        for k, v in extra_subs.items():
            txt = txt.replace(f"[{k}]", str(v))
        return txt

    def _chat_completion(self, messages):
        if not self._client: return None
        try:
            completion = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=0.7
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def pre_optimization_comment(self, n_hypotheses=3):
        if not HAS_OPENAI: return
        prompt = self._fill_prompt(PROMPTS["starter"], {"n_hypotheses": n_hypotheses})
        self._chat_history.append({"role": "user", "content": prompt})
        resp = self._chat_completion(self._chat_history)
        if resp:
            self._chat_history.append({"role": "assistant", "content": resp})
            self._log_to_file(f"PRE-OPTIMIZATION:\n{resp}")

    def comment_and_select_point(self, data: pd.DataFrame, suggestions: pd.DataFrame):
        if not HAS_OPENAI: return
        data_summary = data.tail(5).to_string(index=False)
        sugg_str = suggestions.to_string(index=False)
        prompt = self._fill_prompt(PROMPTS["comment_selection"], {
            "iteration": len(data), "dataset": data_summary, "suggestions": sugg_str
        })
        self._chat_history.append({"role": "user", "content": prompt})
        resp = self._chat_completion(self._chat_history)
        if resp:
            self._chat_history.append({"role": "assistant", "content": resp})
            self._log_to_file(f"ITERATION REVIEW:\n{resp}")
            print(f"\n--- LLM Comment ---\n{resp}\n-------------------")

    def _log_to_file(self, text):
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{text}\n")

# --- BoTorch Optimizer ---
class BoTorchOptimizer:
    def __init__(self, bounds: np.ndarray, random_state: int = 42):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.bounds_np = bounds
        self.bounds_tensor = torch.tensor(bounds.T, device=self.device, dtype=self.dtype)
        torch.manual_seed(random_state)
        self.model = None
        self.train_y_std = None
        self.train_y_mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = torch.tensor(X, device=self.device, dtype=self.dtype)
        self.y_train = torch.tensor(y.reshape(-1, 1), device=self.device, dtype=self.dtype)
        
        self.train_y_mean = self.y_train.mean()
        self.train_y_std = self.y_train.std()
        if self.train_y_std < 1e-9: self.train_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)

        self.train_X_norm = normalize(self.X_train, self.bounds_tensor)
        self.train_y_std_norm = standardize(self.y_train)
        
        self.model = SingleTaskGP(self.train_X_norm, self.train_y_std_norm)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def suggest_batch_qucb(self, q: int = 3, beta: float = 2.0):
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1024]), seed=42)
        qucb = qUpperConfidenceBound(model=self.model, beta=beta, sampler=sampler)
        
        # Constraints logic (Sum of non-P10 <= 5)
        # Note: indices depend on the specific column order.
        idx_p10 = 5
        idx_others = [i for i in range(self.bounds_tensor.shape[1]) if i != idx_p10]
        lower, upper = self.bounds_tensor[0], self.bounds_tensor[1]
        ranges = upper - lower
        sum_lower_others = lower[idx_others].sum()
        rhs_val = (sum_lower_others - 5.0).item()
        coeffs_tensor = -ranges[idx_others]
        indices_tensor = torch.tensor(idx_others, device=self.device, dtype=torch.long)
        
        try:
            candidates_norm, _ = optimize_acqf(
                acq_function=qucb, bounds=torch.stack([torch.zeros_like(lower), torch.ones_like(upper)]),
                q=q, num_restarts=10, raw_samples=512, 
                inequality_constraints=[(indices_tensor, coeffs_tensor, rhs_val)], sequential=True
            )
        except:
             candidates_norm, _ = optimize_acqf(
                acq_function=qucb, bounds=torch.stack([torch.zeros_like(lower), torch.ones_like(upper)]),
                q=q, num_restarts=10, raw_samples=512, sequential=True
            )

        candidates_raw = unnormalize(candidates_norm, self.bounds_tensor).detach().cpu().numpy()
        
        # Rounding and strict constraints
        final_candidates = []
        for cand in candidates_raw:
            final_candidates.append(self._apply_rounding_and_constraints(cand, idx_p10, idx_others))
        return np.array(final_candidates)

    def _apply_rounding_and_constraints(self, candidate, idx_p10, idx_others):
        cand = candidate.copy()
        # Round P10 to 0.2 step, others to 0.25
        p10_val = round(cand[idx_p10] / 0.2) * 0.2
        cand[idx_p10] = max(1.2, p10_val) # Min bound for P10
        cand[idx_others] = np.round(cand[idx_others] / 0.25) * 0.25
        cand[idx_others] = np.maximum(0.0, cand[idx_others])
        
        # Enforce sum <= 5 constraint greedily
        while cand[idx_others].sum() > 5.0 + 1e-6:
            local_indices = np.argsort(cand[idx_others])[::-1]
            reduced = False
            for loc_i in local_indices:
                real_idx = idx_others[loc_i]
                if cand[real_idx] >= 0.25:
                    cand[real_idx] -= 0.25
                    reduced = True
                    break
            if not reduced: break
        return cand

    def predict(self, X: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, device=self.device, dtype=self.dtype)
            X_norm = normalize(X_tensor, self.bounds_tensor)
            posterior = self.model.posterior(X_norm)
            mu = posterior.mean * self.train_y_std + self.train_y_mean
            sigma = posterior.variance.sqrt() * self.train_y_std
            return mu.cpu().numpy().ravel(), sigma.cpu().numpy().ravel()

# --- Visualization ---
def create_enhanced_visualization(suggestions, X, y, feature_cols, output_dir, model_wrapper):
    best_sugg = max(suggestions, key=lambda x: x['predicted_target'])
    best_params = best_sugg['params']
    cols_per_row = 5
    rows_needed = (len(feature_cols) + cols_per_row - 1) // cols_per_row
    
    fig = plt.figure(figsize=(20, 5 * rows_needed + 5))
    gs = gridspec.GridSpec(rows_needed + 1, cols_per_row, height_ratios=[1]*rows_needed + [0.8], hspace=0.4, wspace=0.3)
    
    for i, col_name in enumerate(feature_cols):
        ax = fig.add_subplot(gs[i // cols_per_row, i % cols_per_row])
        x_min, x_max = model_wrapper.bounds_np[i]
        
        # Slice plot
        if x_min == x_max:
             x_grid = np.linspace(x_min - 0.1, x_max + 0.1, 100)
        else:
             x_grid = np.linspace(x_min, x_max, 100)
             
        X_visualize = np.tile(best_params, (100, 1))
        X_visualize[:, i] = x_grid
        y_pred_grid, y_std_grid = model_wrapper.predict(X_visualize)
        
        lower = np.maximum(y_pred_grid - 1.96 * y_std_grid, 0)
        upper = np.maximum(y_pred_grid + 1.96 * y_std_grid, 0)
        
        ax.plot(x_grid, np.maximum(y_pred_grid, 0), color='#2c3e50', linewidth=2.5)
        ax.fill_between(x_grid, lower, upper, color='#3498db', alpha=0.15)
        ax.scatter(X[:, i], y, color='gray', alpha=0.4, s=30)
        ax.scatter([best_params[i]], [best_sugg['predicted_target']], color='#e74c3c', s=150, marker='*', zorder=10)
        ax.set_title(col_name)
        ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'optimization_landscape.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {plot_path}")

# =============================================================================
# PART 4: MAIN WORKFLOW INTEGRATION
# =============================================================================

def run_winter_school_optimization(filepath: str, n_suggestions: int = 12):
    """
    Main entry point for the BoTorch optimization logic.
    """
    output_dir = os.path.dirname(os.path.abspath(filepath))
    
    # 1. Load Data
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Columns specific to Script 3 (Check if these exist, otherwise warn)
    feature_cols = [
        'AcidRed871_0gL', 'L-Cysteine-50gL', 'MethyleneB_250mgL', 'NaCl-3M', 
        'NaOH-1M', 'P10-MIX1', 'PVP-1wt', 'RhodamineB1_0gL', 'SDS-1wt', 'Sodiumsilicate-1wt'
    ]
    target_col = 'Target'
    
    # Check for missing columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing columns in data: {missing}")
        return

    # Preprocessing
    df_clean = df[feature_cols + [target_col]].copy()
    for col in df_clean.columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean.dropna(inplace=True)
    X, y = df_clean[feature_cols].values, df_clean[target_col].values

    # 2. Setup Assistant & Bounds
    mock_params = []
    for i, name in enumerate(feature_cols):
        min_b = max(0, X[:, i].min() * 0.9)
        max_b = max(X[:, i].max() * 1.1, 1e-6)
        # Specific P10 logic
        if i == 5: 
            min_b, max_b, step = 1.0001, max(max_b, 2.0), 0.2
        else:
            step = 0.25
        mock_params.append(MockParameter(name, [min_b, max_b], step))

    mock_exp = MockExperiment("Chemical Opt", mock_params, target_col)
    log_file = os.path.join(output_dir, "assistant_log.md")
    assistant = Assistant(OPENAI_API_KEY, mock_exp, log_file)
    
    print("\n🤖 Assistant Initializing...")
    assistant.pre_optimization_comment()

    # 3. BoTorch Optimization
    print("\nTraining qUCB model...")
    bounds = np.array([p.bounds for p in mock_params])
    opt = BoTorchOptimizer(bounds=bounds)
    opt.fit(X, y)
    
    print(f"Generating {n_suggestions} suggestions...")
    candidates = opt.suggest_batch_qucb(q=n_suggestions)
    
    # 4. Format Results
    results = []
    for cand in candidates:
        mu, std = opt.predict(cand.reshape(1, -1))
        results.append({
            'params': cand, 
            'predicted_target': float(mu[0]), 
            'uncertainty': float(std[0])
        })
    
    res_df = pd.DataFrame([
        {**{feature_cols[i]: r['params'][i] for i in range(len(feature_cols))},
         'Predicted Target': r['predicted_target'], 
         'Uncertainty': r['uncertainty']} 
        for r in results
    ])
    
    out_file = os.path.join(output_dir, 'optimization_suggestions.xlsx')
    res_df.to_excel(out_file, index=False)
    print(f"✓ Suggestions saved to: {out_file}")

    # 5. Review & Visualize
    assistant.comment_and_select_point(df_clean, res_df)
    create_enhanced_visualization(results, X, y, feature_cols, output_dir, opt)
    
    return out_file

# =============================================================================
# MAIN EXECUTION MENU
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("      SCIENTIFIC OPTIMIZATION & PREDICTION TOOL      ")
    print("="*60)
    print("1. Train Standard GP Model (replicates Script 2)")
    print("2. Run BoTorch Optimization with LLM (replicates Script 3)")
    print("3. Predict using Saved GP Model (replicates Script 1)")
    print("4. Full Workflow (Train GP -> Optimize -> Cross-Check)")
    
    choice = input("\nEnter choice (1-4): ")

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    if choice == "1":
        csv_in = input("Enter training CSV path (default: datafull.csv): ") or "datafull.csv"
        train_gp_on_file(csv_in)

    elif choice == "2":
        excel_in = input("Enter data Excel path (default: HER_virtual_data.xlsx): ") or "HER_virtual_data.xlsx"
        if os.path.exists(excel_in):
            run_winter_school_optimization(excel_in)
        else:
            print(f"❌ File {excel_in} not found.")

    elif choice == "3":
        excel_in = input("Enter Excel file to predict (default: optimization_suggestions.xlsx): ") or "optimization_suggestions.xlsx"
        predict_from_excel_file(excel_in)

    elif choice == "4":
        # Full Sequence
        # 1. Train GP on historical data
        train_gp_on_file("datafull.csv")
        
        # 2. Optimize to get new suggestions
        suggestions_file = "optimization_suggestions.xlsx"
        if os.path.exists("HER_virtual_data.xlsx"):
            suggestions_file = run_winter_school_optimization("HER_virtual_data.xlsx")
        
        # 3. Predict on suggestions using the GP from step 1
        if suggestions_file and os.path.exists(suggestions_file):
            print("\n--- Cross-Checking Suggestions with Standard GP ---")
            predict_from_excel_file(suggestions_file, output_file="final_verified_suggestions.xlsx")
            
    else:
        print("Invalid choice.")
