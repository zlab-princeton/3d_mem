import numpy as np
from scipy import linalg


def compute_mu_sigma(feats: np.ndarray):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError(f"mean shape mismatch {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"cov shape mismatch {sigma1.shape} vs {sigma2.shape}")
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # jitter both covs if product gets singular
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
