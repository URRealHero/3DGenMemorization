#!/usr/bin/env python3
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

# ==========================================
# 1. Exact Reference Implementation
# ==========================================
def Zu_from_distances(LPn, LQm): 
    """
    Calculates Z_u using pre-computed distances.
    Logic taken directly from the reference implementation (Zu function).
    
    Inputs: 
        LPn: Distance to training NN for test set (Real)
        LQm: Distance to training NN for generated set (Fake)
    """
    # Ensure inputs are flat arrays
    LPn = np.array(LPn)
    LQm = np.array(LQm)

    m = LQm.shape[0]
    n = LPn.shape[0]

    if m == 0 or n == 0:
        raise ValueError("Input arrays cannot be empty.")

    # Get Mann-Whitney U score and manually Z-score it using the conditions of null hypothesis H_0 
    # Reference: u, _ = mannwhitneyu(LQm, LPn, alternative = 'less')
    u, _ = mannwhitneyu(LQm, LPn, alternative='less')
    
    mean = (n * m / 2) - 0.5 # 0.5 is continuity correction
    std = np.sqrt(n * m * (n + m + 1) / 12)
    
    Z_u = (u - mean) / std 
    return Z_u

# ==========================================
# 2. JSON Loading Logic
# ==========================================
def extract_distances(json_path: Path):
    """
    Extracts 'distance' values from the user's specific JSON format.
    Returns a list of floats.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    distances = []
    
    # Check if data is grouped by model (dict of dicts) or flat (dict of entries)
    # Heuristic: check if the first value is a dict that does NOT have 'distance' key directly
    first_val = next(iter(data.values())) if data else {}
    is_grouped = isinstance(first_val, dict) and "distance" not in first_val

    if is_grouped:
        # Iterate over models, then queries
        for model_res in data.values():
            for entry in model_res.values():
                # Entry might be list [{...}] or dict {...}
                item = entry[0] if isinstance(entry, list) else entry
                if item and "distance" in item:
                    distances.append(float(item["distance"]))
    else:
        # Flat dictionary (Query ID -> Result)
        for entry in data.values():
            item = entry[0] if isinstance(entry, list) else entry
            if item and "distance" in item:
                distances.append(float(item["distance"]))

    return distances

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_json", type=Path, help="Path to generated set retrieval JSON (L(Qm))")
    parser.add_argument("test_json", type=Path, help="Path to test set retrieval JSON (L(Pn))")
    args = parser.parse_args()

    # Load distances (L = Length/Distance to Training Set)
    LQm = extract_distances(args.gen_json)
    LPn = extract_distances(args.test_json)

    print(f"Loaded L(Qm) [Generated]: {len(LQm)} samples")
    print(f"Loaded L(Pn) [Test]     : {len(LPn)} samples")

    # Calculate
    score = Zu_from_distances(LPn, LQm)

    print("-" * 30)
    print(f"Z_u Score: {score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()