import argparse
import numpy as np
from pathlib import Path
from numpy.linalg import eigh

def cov_biased(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False, ddof=0).astype(np.float64)

def sqrtm_spd(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, V = eigh((C + C.T) * 0.5)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T

def fd_gaussian(m1, C1, m2, C2, eps: float = 1e-12) -> float:
    m1 = m1.astype(np.float64); m2 = m2.astype(np.float64)
    C1 = C1.astype(np.float64); C2 = C2.astype(np.float64)
    diff2 = float(np.dot(m1 - m2, m1 - m2))
    S1 = sqrtm_spd(C1, eps=eps)
    M = S1 @ C2 @ S1
    S = sqrtm_spd(M, eps=eps)
    return diff2 + float(np.trace(C1 + C2 - 2.0 * S))

def load_array(path: Path) -> np.ndarray:
    if not path.exists(): raise FileNotFoundError(f"{path} missing")
    if path.suffix == ".npy": return np.load(path)
    with np.load(path) as z:
        for k in ["arr","X","features","emb","embeddings","data"]:
            if k in z: return z[k]
        return z[z.files[0]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path, help="Reference .npz/.npy")
    p.add_argument("gen", type=Path, help="Generated .npz/.npy")
    args = p.parse_args()

    X_ref = load_array(args.ref).astype(np.float64)
    X_gen = load_array(args.gen).astype(np.float64)

    if X_ref.shape[1] != X_gen.shape[1]:
        raise ValueError(f"Dim mismatch: {X_ref.shape[1]} vs {X_gen.shape[1]}")

    # 2. Statistics (you can whiten ref data before saving npz if needed)
    m_ref, C_ref = X_ref.mean(axis=0), cov_biased(X_ref)
    m_gen, C_gen = X_gen.mean(axis=0), cov_biased(X_gen)

    score = fd_gaussian(m_ref, C_ref, m_gen, C_gen)
    
    print(f"FPD: {score:.6f}")

if __name__ == "__main__":
    main()