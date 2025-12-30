import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

def Zu_from_distances(LPn, LQm): 
    """
    Calculates Z_u using pre-computed distances.
    
    LPn: Distance to training NN for test set
    LQm: Distance to training NN for generated set
    """
    LPn = np.array(LPn)
    LQm = np.array(LQm)

    m = LQm.shape[0]
    n = LPn.shape[0]

    if m == 0 or n == 0:
        raise ValueError("Input arrays cannot be empty.")

    u, _ = mannwhitneyu(LQm, LPn, alternative='less')
    
    mean = (n * m / 2) - 0.5 # 0.5 is continuity correction
    std = np.sqrt(n * m * (n + m + 1) / 12)
    
    Z_u = (u - mean) / std 
    return Z_u

def extract_distances(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    distances = []
    
    first_val = next(iter(data.values())) if data else {}
    is_grouped = isinstance(first_val, dict) and "distance" not in first_val

    if is_grouped:
        for model_res in data.values():
            for entry in model_res.values():
                item = entry[0] if isinstance(entry, list) else entry
                if item and "distance" in item:
                    distances.append(float(item["distance"]))
    else:
        for entry in data.values():
            item = entry[0] if isinstance(entry, list) else entry
            if item and "distance" in item:
                distances.append(float(item["distance"]))

    return distances

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_json", type=Path, help="Path to generated set retrieval JSON (L(Qm))")
    parser.add_argument("test_json", type=Path, help="Path to test set retrieval JSON (L(Pn))")
    args = parser.parse_args()

    # Load distances to training set
    LQm = extract_distances(args.gen_json)
    LPn = extract_distances(args.test_json)

    print(f"Loaded L(Qm) [Generated]: {len(LQm)} samples")
    print(f"Loaded L(Pn) [Test]     : {len(LPn)} samples")

    score = Zu_from_distances(LPn, LQm)

    print("-" * 30)
    print(f"Z_u Score: {score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()