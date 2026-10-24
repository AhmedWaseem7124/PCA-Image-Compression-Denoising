"""
=============================================================================
 day04_pca_image_compression_denoising.py
 PCA via SVD from Scratch -- Image Compression & Spatial Denoising
=============================================================================
 Author  : ML/CV Engineer
 Date    : 2026-10-24
 Dataset : fetch_olivetti_faces (400 x 64x64 grayscale faces)
           Offline fallback -> load_digits upsampled  ->  synthetic faces
 Engine  : Pure NumPy (np.linalg.svd)  -- NO scikit-learn PCA used
=============================================================================
"""

# --------------------------- Imports ----------------------------------------
import sys
# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Python < 3.7 fallback
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")

# -------------------------- Constants ---------------------------------------
RANDOM_SEED   = 42
rng           = np.random.default_rng(RANDOM_SEED)

NOISE_SIGMA   = 0.25
K_COMPRESSION = [5, 20, 50, 150]
K_DENOISE     = 40
VAR_THRESHOLDS = [0.80, 0.90, 0.95]

ARTIFACT_PATH = "day04_pca_artifact.png"
IMG_SIZE      = 64          # target pixel width/height

# Dark theme colours
DARK_BG       = "#0d1117"
PANEL_BG      = "#161b22"
ACCENT        = "#58a6ff"
ACCENT2       = "#3fb950"
ACCENT3       = "#f78166"
ACCENT4       = "#d2a8ff"
TEXT_COL      = "#e6edf3"
GRID_COL      = "#30363d"

# ===========================================================================
#  1.  DATASET LOADING  (Olivetti -> digits fallback -> synthetic fallback)
# ===========================================================================

def _resize_image(img_2d: np.ndarray, target: int) -> np.ndarray:
    """Nearest-neighbour resize of a 2-D grayscale image."""
    h, w = img_2d.shape
    row_idx = (np.arange(target) * h / target).astype(int)
    col_idx = (np.arange(target) * w / target).astype(int)
    return img_2d[np.ix_(row_idx, col_idx)]


def _make_synthetic_faces(n: int = 400, size: int = 64) -> np.ndarray:
    """Generate structured synthetic 'faces' as a reliable last-resort fallback."""
    local_rng = np.random.default_rng(RANDOM_SEED)
    faces = []
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    for i in range(n):
        # Randomised Gaussian blobs mimicking facial structure
        cx  = local_rng.uniform(-0.2, 0.2)
        cy  = local_rng.uniform(-0.1, 0.2)
        r   = local_rng.uniform(0.5, 0.8)
        face = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2))
        # Eyes
        for ex in [-0.25, 0.25]:
            ey = 0.15 + local_rng.uniform(-0.05, 0.05)
            er = local_rng.uniform(0.06, 0.10)
            face -= 0.4 * np.exp(-((xx - ex)**2 + (yy - ey)**2) / (2 * er**2))
        # Nose
        face += 0.2 * np.exp(-((xx)**2 + (yy + 0.05)**2) / 0.01)
        # Mouth
        mx = local_rng.uniform(-0.1, 0.1)
        face -= 0.3 * np.exp(-((xx - mx)**2 + (yy + 0.3)**2) / 0.015)
        # Normalise to [0, 1]
        face = (face - face.min()) / (face.max() - face.min() + 1e-9)
        face += local_rng.normal(0, 0.03, face.shape)
        face = np.clip(face, 0, 1)
        faces.append(face.ravel())
    return np.array(faces, dtype=np.float64)


def load_dataset() -> tuple[np.ndarray, int, str]:
    """
    Returns
    -------
    X        : (N, D) float64 array with pixel values in [0, 1]
    img_size : pixel dimension (img_size x img_size)
    source   : human-readable string naming the dataset used
    """
    # -- Try Olivetti ------------------------------------------------------
    try:
        from sklearn.datasets import fetch_olivetti_faces
        print("  [INFO] Fetching Olivetti Faces dataset ...", flush=True)
        data   = fetch_olivetti_faces(shuffle=True, random_state=RANDOM_SEED)
        X      = data.images                          # (400, 64, 64) float64
        X_flat = X.reshape(X.shape[0], -1)            # (400, 4096)
        print(f"  [OK]   Olivetti loaded  -> shape {X_flat.shape}", flush=True)
        return X_flat, 64, "Olivetti Faces (400 x 64x64)"
    except Exception as e:
        print(f"  [WARN] Olivetti unavailable: {e}", flush=True)

    # -- Try load_digits upsampled to 64x64 -------------------------------
    try:
        from sklearn.datasets import load_digits
        print("  [INFO] Falling back to load_digits (upsampled 64x64) ...", flush=True)
        digits   = load_digits()
        X_digits = digits.images.astype(np.float64)   # (1797, 8, 8)
        X_digits /= X_digits.max()
        upsampled = np.array(
            [_resize_image(img, IMG_SIZE).ravel() for img in X_digits]
        )
        print(f"  [OK]   Digits (upsampled) -> shape {upsampled.shape}", flush=True)
        return upsampled, IMG_SIZE, "load_digits upsampled to 64x64"
    except Exception as e:
        print(f"  [WARN] load_digits unavailable: {e}", flush=True)

    # -- Synthetic fallback ------------------------------------------------
    print("  [INFO] Using synthetic structured faces ...", flush=True)
    X_syn = _make_synthetic_faces(n=400, size=IMG_SIZE)
    print(f"  [OK]   Synthetic faces     -> shape {X_syn.shape}", flush=True)
    return X_syn, IMG_SIZE, "Synthetic Structured Faces (400 x 64x64)"


# ===========================================================================
#  2.  FROM-SCRATCH PCA ENGINE
# ===========================================================================

class ScratchPCA:
    """
    Principal Component Analysis implemented in pure NumPy via full SVD.

    Mathematical convention
    -----------------------
    X_centered  = X - mu                        shape (N, D)
    SVD         : X_centered = U Sum V^T         shapes (N,N), (N,), (D,D) [full]
                              or economy SVD   (N,K), (K,), (K,D)
    lambda_i         = sigma_i^2 / (N - 1)              eigenvalue of covariance matrix
    EVR_i       = lambda_i / Sum lambda_j                 explained variance ratio

    Projection  : Z_k = X_centered @ V_k       (N, k)
    Reconstruct : X_k = Z_k @ V_k^T + mu       (N, D)
    """

    def __init__(self):
        self.mean_       : np.ndarray | None = None   # (D,)
        self.components_ : np.ndarray | None = None   # (D, D) or (min(N,D), D)
        self.singular_values_ : np.ndarray | None = None
        self.explained_variance_       : np.ndarray | None = None
        self.explained_variance_ratio_ : np.ndarray | None = None
        self.cumulative_variance_ratio_: np.ndarray | None = None
        self._n_samples  : int = 0

    # -- fit ---------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "ScratchPCA":
        """
        Fit PCA to the (N, D) data matrix X.
        Internally runs economy SVD (full_matrices=False) on X_centered.
        """
        N, D = X.shape
        self._n_samples = N

        # 1. Mean face / pixel mean
        self.mean_ = X.mean(axis=0)                   # (D,)

        # 2. Mean-centre
        Xc = X - self.mean_                           # (N, D)

        # 3. Economy SVD  ->  Xc = U Sum V^T
        #    V rows are RIGHT singular vectors == principal components
        print("  [INFO] Running economy SVD ...", flush=True)
        U, sigma, Vt = np.linalg.svd(Xc, full_matrices=False)
        # shapes: U (N, K), sigma (K,), Vt (K, D)  where K = min(N,D)

        self.singular_values_  = sigma                 # (K,)
        self.components_       = Vt                    # (K, D)  <- eigenvectors as rows

        # 4. Eigenvalues and explained variance
        lambdas = sigma**2 / (N - 1)                  # eigenvalues of cov matrix
        total   = lambdas.sum()
        self.explained_variance_       = lambdas
        self.explained_variance_ratio_ = lambdas / total
        self.cumulative_variance_ratio_= np.cumsum(self.explained_variance_ratio_)

        print(f"  [OK]   SVD complete. "
              f"Top-10 singular values: {np.round(sigma[:10], 2)}", flush=True)
        return self

    # -- project -----------------------------------------------------------
    def project(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Project X onto the top-k principal components.

        Parameters
        ----------
        X : (N, D)
        k : number of components

        Returns
        -------
        Z_k : (N, k)   low-dimensional coordinates
        """
        if k > self.components_.shape[0]:
            raise ValueError(f"k={k} exceeds available components ({self.components_.shape[0]})")
        Xc  = X - self.mean_                          # (N, D)
        V_k = self.components_[:k].T                  # (D, k)
        return Xc @ V_k                               # (N, k)

    # -- reconstruct -------------------------------------------------------
    def reconstruct(self, Z_k: np.ndarray, k: int) -> np.ndarray:
        """
        Reconstruct data from low-dimensional codes Z_k.

        Parameters
        ----------
        Z_k : (N, k)
        k   : must match Z_k.shape[1]

        Returns
        -------
        X_hat : (N, D)
        """
        V_k = self.components_[:k].T                  # (D, k)
        return Z_k @ V_k.T + self.mean_               # (N, D)

    # -- helpers -----------------------------------------------------------
    def n_components_for_variance(self, threshold: float) -> int:
        """Return the minimum k such that cumulative EVR >= threshold."""
        idx = np.searchsorted(self.cumulative_variance_ratio_, threshold)
        return int(idx) + 1

    def compression_ratio(self, k: int, D: int, N: int) -> float:
        """
        Memory compression ratio = original_floats / compressed_floats.
        Original  : N x D
        Compressed: N x k  (scores) + k x D  (components) + D (mean)
        """
        original   = N * D
        compressed = N * k + k * D + D
        return original / compressed


# ===========================================================================
#  3.  METRICS
# ===========================================================================

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse_val = mse(original, reconstructed)
    if mse_val < 1e-15:
        return float("inf")
    return 10.0 * np.log10(max_val**2 / mse_val)


# ===========================================================================
#  4.  EXPERIMENTS
# ===========================================================================

def run_compression_experiment(
    pca: ScratchPCA,
    X_test: np.ndarray,
    k_list: list[int],
    N_train: int,
    D: int,
    idx: int = 0,
) -> dict:
    """
    Experiment A: Compress a single test image with different k values.

    Returns
    -------
    results : dict keyed by k  ->  {mse, compression_ratio, reconstructed}
    """
    results = {}
    x_single = X_test[[idx]]   # keep 2-D shape (1, D)
    for k in k_list:
        Z_k   = pca.project(x_single, k)
        X_hat = pca.reconstruct(Z_k, k)
        X_hat = np.clip(X_hat, 0, 1)
        results[k] = {
            "mse"              : mse(x_single, X_hat),
            "compression_ratio": pca.compression_ratio(k, D, N_train),
            "reconstructed"    : X_hat.reshape(-1),   # (D,)
        }
    results["original"] = x_single.reshape(-1)
    return results


def run_denoising_experiment(
    pca: ScratchPCA,
    X_test: np.ndarray,
    k: int,
    noise_sigma: float,
    idx: int = 0,
) -> dict:
    """
    Experiment B: Denoise a test image via PCA projection / reconstruction.

    Returns
    -------
    results : dict with clean, noisy, reconstructed, residual, metrics
    """
    x_clean = X_test[[idx]]                                      # (1, D)
    noise   = rng.normal(0, noise_sigma, x_clean.shape)
    x_noisy = np.clip(x_clean + noise, 0, 1)

    Z_k     = pca.project(x_noisy, k)
    x_recon = np.clip(pca.reconstruct(Z_k, k), 0, 1)

    residual = np.abs(x_noisy - x_recon)

    return {
        "clean"      : x_clean.reshape(-1),
        "noisy"      : x_noisy.reshape(-1),
        "reconstructed": x_recon.reshape(-1),
        "residual"   : residual.reshape(-1),
        "mse_noisy"  : mse(x_clean, x_noisy),
        "mse_recon"  : mse(x_clean, x_recon),
        "psnr_noisy" : psnr(x_clean, x_noisy),
        "psnr_recon" : psnr(x_clean, x_recon),
    }


# ===========================================================================
#  5.  VISUALISATION  (dark-themed 2x2 dashboard)
# ===========================================================================

def _apply_dark_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)


def plot_panel1_variance(ax, pca: ScratchPCA):
    """Panel 1 -- Cumulative Explained Variance Curve."""
    n_comp = len(pca.cumulative_variance_ratio_)
    ks     = np.arange(1, n_comp + 1)
    cvr    = pca.cumulative_variance_ratio_ * 100

    ax.fill_between(ks, cvr, alpha=0.12, color=ACCENT)
    ax.plot(ks, cvr, color=ACCENT, linewidth=1.8, label="Cumulative EVR")

    threshold_colors = [ACCENT3, ACCENT2, ACCENT4]
    for thr, col in zip(VAR_THRESHOLDS, threshold_colors):
        k_thr = pca.n_components_for_variance(thr)
        ax.axhline(thr * 100, color=col, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(k_thr,    color=col, linestyle=":",  linewidth=1.0, alpha=0.8)
        ax.annotate(
            f"{int(thr*100)}% @ k={k_thr}",
            xy=(k_thr, thr * 100),
            xytext=(k_thr + max(n_comp * 0.03, 5), thr * 100 - 3),
            color=col, fontsize=7,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

    _apply_dark_style(
        ax,
        title="Panel 1 -- Cumulative Explained Variance vs. k",
        xlabel="Number of Principal Components  k",
        ylabel="Cumulative Explained Variance (%)",
    )
    ax.set_xlim(0, min(n_comp, 400))
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.7)


def plot_panel2_eigenfaces(ax, pca: ScratchPCA, img_size: int):
    """Panel 2 -- Mean Image + top-5 Eigenfaces."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(
        "Panel 2 -- Mean Face & Top-5 Eigenfaces",
        color=TEXT_COL, fontsize=10, fontweight="bold", pad=8,
    )
    ax.axis("off")

    n_show = 6          # mean + 5 eigenfaces
    fig    = ax.figure
    pos    = ax.get_position()

    # Carve sub-axes inside Panel 2
    padding = 0.005
    w_each  = (pos.width - padding * (n_show + 1)) / n_show
    h_each  = pos.height * 0.72
    y_start = pos.y0 + pos.height * 0.12

    labels   = ["Mean"] + [f"PC {i+1}" for i in range(5)]
    images_2d = [pca.mean_.reshape(img_size, img_size)] + [
        pca.components_[i].reshape(img_size, img_size) for i in range(5)
    ]

    sub_axes = []
    for j in range(n_show):
        x0   = pos.x0 + padding + j * (w_each + padding)
        sub_ax = fig.add_axes([x0, y_start, w_each, h_each])
        sub_ax.set_facecolor(DARK_BG)
        img  = images_2d[j]
        cmap = "gray" if j == 0 else "RdBu_r"
        sub_ax.imshow(img, cmap=cmap, aspect="equal",
                      vmin=img.min(), vmax=img.max())
        sub_ax.axis("off")

        # Label below
        label_ax = fig.add_axes([x0, y_start - 0.022, w_each, 0.018])
        label_ax.set_facecolor(PANEL_BG)
        label_ax.axis("off")
        evr = pca.explained_variance_ratio_[j - 1] * 100 if j > 0 else None
        caption = labels[j] if j == 0 else f"{labels[j]}\n{evr:.1f}% var"
        label_ax.text(
            0.5, 0.5, caption,
            ha="center", va="center",
            fontsize=6.5, color=TEXT_COL,
            transform=label_ax.transAxes,
        )
        sub_axes.append(sub_ax)

    # Separator line
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)


def plot_panel3_compression(
    ax, comp_results: dict, img_size: int, k_list: list[int]
):
    """Panel 3 -- Progressive Image Compression."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(
        "Panel 3 -- Progressive Image Compression",
        color=TEXT_COL, fontsize=10, fontweight="bold", pad=8,
    )
    ax.axis("off")

    fig   = ax.figure
    pos   = ax.get_position()
    n_img = len(k_list) + 1    # k values + original

    padding = 0.004
    w_each  = (pos.width - padding * (n_img + 1)) / n_img
    h_each  = pos.height * 0.72
    y_start = pos.y0 + pos.height * 0.12

    items = [("Original\n4096-D", comp_results["original"])] + [
        (
            f"k={k}\nMSE={comp_results[k]['mse']:.4f}\n"
            f"CR={comp_results[k]['compression_ratio']:.1f}x",
            comp_results[k]["reconstructed"],
        )
        for k in k_list
    ]

    for j, (label, img_flat) in enumerate(items):
        x0 = pos.x0 + padding + j * (w_each + padding)
        sub_ax = fig.add_axes([x0, y_start, w_each, h_each])
        sub_ax.set_facecolor(DARK_BG)
        sub_ax.imshow(img_flat.reshape(img_size, img_size), cmap="gray",
                      vmin=0, vmax=1, aspect="equal")
        sub_ax.axis("off")

        label_ax = fig.add_axes([x0, y_start - 0.030, w_each, 0.028])
        label_ax.set_facecolor(PANEL_BG)
        label_ax.axis("off")
        col = ACCENT if j == 0 else ACCENT2
        label_ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=6.0, color=col,
            transform=label_ax.transAxes,
            multialignment="center",
        )


def plot_panel4_denoising(ax, denoise_results: dict, img_size: int, k: int):
    """Panel 4 -- Denoising Comparison."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(
        "Panel 4 -- Image Denoising via PCA",
        color=TEXT_COL, fontsize=10, fontweight="bold", pad=8,
    )
    ax.axis("off")

    fig   = ax.figure
    pos   = ax.get_position()
    n_img = 4
    padding = 0.004
    w_each  = (pos.width - padding * (n_img + 1)) / n_img
    h_each  = pos.height * 0.72
    y_start = pos.y0 + pos.height * 0.12

    dr = denoise_results
    items = [
        ("Clean\nOriginal",         dr["clean"],        "gray",   0, 1),
        ("Noisy Input\n"
         f"PSNR={dr['psnr_noisy']:.1f}dB",
                                    dr["noisy"],        "gray",   0, 1),
        (f"PCA Recon\n(k={k})\n"
         f"PSNR={dr['psnr_recon']:.1f}dB",
                                    dr["reconstructed"],"gray",   0, 1),
        ("Residual\n|Noisy-Recon|", dr["residual"],     "hot",    None, None),
    ]
    label_colors = [ACCENT, ACCENT3, ACCENT2, ACCENT4]

    for j, (label, img_flat, cmap, vmin, vmax) in enumerate(items):
        x0 = pos.x0 + padding + j * (w_each + padding)
        sub_ax = fig.add_axes([x0, y_start, w_each, h_each])
        sub_ax.set_facecolor(DARK_BG)
        kwargs = dict(cmap=cmap, aspect="equal")
        if vmin is not None:
            kwargs.update(vmin=vmin, vmax=vmax)
        sub_ax.imshow(img_flat.reshape(img_size, img_size), **kwargs)
        sub_ax.axis("off")

        label_ax = fig.add_axes([x0, y_start - 0.030, w_each, 0.028])
        label_ax.set_facecolor(PANEL_BG)
        label_ax.axis("off")
        label_ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=6.0, color=label_colors[j],
            transform=label_ax.transAxes,
            multialignment="center",
        )


def build_dashboard(
    pca: ScratchPCA,
    comp_results: dict,
    denoise_results: dict,
    img_size: int,
    dataset_name: str,
    k_list: list[int],
    k_denoise: int,
    save_path: str,
):
    """Assemble the 2x2 dark-themed dashboard and save to PNG."""
    plt.rcParams.update({
        "font.family"     : "DejaVu Sans",
        "text.color"      : TEXT_COL,
        "axes.facecolor"  : PANEL_BG,
        "figure.facecolor": DARK_BG,
    })

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)

    # Outer title
    fig.suptitle(
        f"PCA via SVD -- Image Compression & Denoising\nDataset: {dataset_name}",
        fontsize=14, fontweight="bold", color=TEXT_COL, y=0.98,
    )

    # 2x2 grid with extra space between rows for sub-image labels
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.04, right=0.97,
        top=0.93, bottom=0.04,
        hspace=0.35, wspace=0.22,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_panel1_variance(ax1, pca)
    plot_panel2_eigenfaces(ax2, pca, img_size)
    plot_panel3_compression(ax3, comp_results, img_size, k_list)
    plot_panel4_denoising(ax4, denoise_results, img_size, k_denoise)

    plt.savefig(save_path, dpi=160, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"\n  [SAVED] Dashboard -> {save_path}", flush=True)


# ===========================================================================
#  6.  TERMINAL SUMMARY TABLE
# ===========================================================================

def print_summary_table(
    pca: ScratchPCA,
    comp_results: dict,
    denoise_results: dict,
    k_list: list[int],
    k_denoise: int,
    D: int,
    N: int,
):
    DIVIDER  = "-" * 85
    HDIVIDER = "=" * 85

    print()
    print(HDIVIDER)
    print("  PCA SVD  --  Experiment Summary")
    print(HDIVIDER)

    # -- Variance thresholds -----------------------------------------------
    print()
    print("  Variance Thresholds")
    print(f"  {'Threshold':>10}   {'Components k':>14}")
    print("  " + "-" * 28)
    for thr in VAR_THRESHOLDS:
        k_thr = pca.n_components_for_variance(thr)
        print(f"  {thr*100:>9.0f}%   {k_thr:>14d}")

    # -- Compression Experiment A ------------------------------------------
    print()
    print("  Experiment A -- Image Compression")
    header = (
        f"  {'k':>6}  {'Cum. Var %':>12}  {'Comp. Ratio':>13}"
        f"  {'MSE':>10}  {'Bits Saved':>10}"
    )
    print(header)
    print("  " + DIVIDER[:len(header) - 2])

    for k in k_list:
        cum_var = pca.cumulative_variance_ratio_[k - 1] * 100
        cr      = comp_results[k]["compression_ratio"]
        err     = comp_results[k]["mse"]
        savings = (1 - 1 / cr) * 100
        print(
            f"  {k:>6}  {cum_var:>11.2f}%  {cr:>11.2f}x"
            f"  {err:>10.6f}  {savings:>9.1f}%"
        )

    # -- Denoising Experiment B --------------------------------------------
    print()
    print(f"  Experiment B -- Image Denoising  (k = {k_denoise}, sigma_noise = {NOISE_SIGMA})")
    print(f"  {'Metric':>14}  {'Noisy Input':>13}  {'PCA Recon':>12}  {'Gain':>8}")
    print("  " + "-" * 55)

    mse_noisy  = denoise_results["mse_noisy"]
    mse_recon  = denoise_results["mse_recon"]
    psnr_noisy = denoise_results["psnr_noisy"]
    psnr_recon = denoise_results["psnr_recon"]

    print(f"  {'MSE':>14}  {mse_noisy:>13.6f}  {mse_recon:>12.6f}"
          f"  {mse_noisy - mse_recon:>+8.6f}")
    print(f"  {'PSNR (dB)':>14}  {psnr_noisy:>13.2f}  {psnr_recon:>12.2f}"
          f"  {psnr_recon - psnr_noisy:>+8.2f}")

    print()
    print(HDIVIDER)
    print()


# ===========================================================================
#  7.  MAIN
# ===========================================================================

def main():
    print()
    print("=" * 60)
    print("  Day 04 -- PCA Image Compression & Denoising")
    print("  Pure NumPy SVD Engine | Dark Dashboard")
    print("=" * 60)

    # -- 1. Load data ------------------------------------------------------
    print("\n[1/4] Loading dataset ...")
    X, img_size, dataset_name = load_dataset()
    N, D = X.shape
    print(f"       N={N}  D={D}  img={img_size}x{img_size}")

    # Train / test split (manual, no sklearn)
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    idx_all   = rng_split.permutation(N)
    n_test    = max(20, int(0.1 * N))
    test_idx  = idx_all[:n_test]
    train_idx = idx_all[n_test:]

    X_train = X[train_idx]
    X_test  = X[test_idx]
    print(f"       Train: {X_train.shape}   Test: {X_test.shape}")

    # -- 2. Fit PCA --------------------------------------------------------
    print("\n[2/4] Fitting ScratchPCA ...")
    pca = ScratchPCA()
    pca.fit(X_train)

    # -- 3. Experiments ----------------------------------------------------
    print("\n[3/4] Running experiments ...")

    test_face_idx = 0   # index within X_test

    # Valid k_list: cap at available components
    max_k         = pca.components_.shape[0]
    k_list_valid  = [k for k in K_COMPRESSION if k <= max_k]
    if not k_list_valid:
        k_list_valid = [min(5, max_k), min(20, max_k),
                        min(50, max_k), min(150, max_k)]
        k_list_valid = sorted(set(k_list_valid))

    k_denoise_valid = min(K_DENOISE, max_k)

    print(f"  Compression k list : {k_list_valid}")
    print(f"  Denoising k        : {k_denoise_valid}")

    comp_results = run_compression_experiment(
        pca, X_test, k_list_valid, N_train=len(X_train), D=D, idx=test_face_idx
    )

    denoise_results = run_denoising_experiment(
        pca, X_test, k=k_denoise_valid, noise_sigma=NOISE_SIGMA, idx=test_face_idx
    )

    # -- 4. Visualise ------------------------------------------------------
    print("\n[4/4] Building dashboard ...")
    build_dashboard(
        pca            = pca,
        comp_results   = comp_results,
        denoise_results= denoise_results,
        img_size       = img_size,
        dataset_name   = dataset_name,
        k_list         = k_list_valid,
        k_denoise      = k_denoise_valid,
        save_path      = ARTIFACT_PATH,
    )

    # -- Terminal summary --------------------------------------------------
    print_summary_table(
        pca            = pca,
        comp_results   = comp_results,
        denoise_results= denoise_results,
        k_list         = k_list_valid,
        k_denoise      = k_denoise_valid,
        D              = D,
        N              = N,
    )

    print(f"  Done!  Artifact saved to -> {ARTIFACT_PATH}")
    print()


if __name__ == "__main__":
    main()
