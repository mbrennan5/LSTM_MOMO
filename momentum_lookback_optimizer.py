# ==============================================================================
# momentum_lookback_optimizer.py — Joint Lookback × Z-Lens Optimizer
# Momentum Edition  |  adapted from ta_feature_factory.py (Trend Edition)
#
# What this adds vs the Trend Optimizer:
#   1. MOMENTUM_FEATURE_CATALOG — 26 momentum indicators numbered for selection
#   2. generate_momentum_features_v2(df, indicator_n, z_n, selected)
#      indicator_n scales ALL momentum windows (ROC periods, MHL, Hurst, etc.)
#      z_n is the SINGLE lens window applied to selected seeds
#   3. Exhaustive grid search over the (indicator_n × z_n) joint space
#   4. Final sovereign audit at the optimal pair (ledger + BRAIN_LOCKS sync)
#
# Mandate (Sovereign framework):
#   Never optimise the seed in isolation.  indicator_n and z_n are COUPLED.
#   Grid search finds the joint optimum; GA-style filtering is BRAIN_LOCKS.
# ==============================================================================
from __future__ import annotations
import os, gc, warnings, datetime, random, itertools, logging
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('peewee').setLevel(logging.CRITICAL)

# ── Colab / local detection ───────────────────────────────────────────────────
try:
    from google.colab import drive as _colab_drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import tensorflow as tf


# ==============================================================================
# BLOCK 0: GPU SETUP
# ==============================================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU — running on CPU.")
        return False
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"✅ Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    with tf.device('/device:GPU:0'):
        _ = tf.random.normal((10, 10)) @ tf.random.normal((10, 10))
    print(f"✅ GPU confirmed: {tf.test.gpu_device_name()}")
    return True

GPU_AVAILABLE = setup_gpu()
DEVICE        = '/device:GPU:0' if GPU_AVAILABLE else '/cpu:0'
print(f"[SYSTEM] Active compute device: {DEVICE}\n")


# ==============================================================================
# BLOCK 1: SYSTEM INITIALISATION
# ==============================================================================
import numpy as np, pandas as pd, yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import GRU, LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from numba import jit
from datetime import datetime as _dt
try:
    from scipy.signal import welch as _scipy_welch
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Output directory ───────────────────────────────────────────────────────────
if IN_COLAB:
    if not os.path.exists('/content/drive'):
        _colab_drive.mount('/content/drive', force_remount=True)
    _BASE_OUT = '/content/drive/MyDrive/judicial_results'
else:
    _BASE_OUT = os.path.join(os.path.dirname(__file__), 'judicial_results')

TEST_NAME  = "Momentum_FeatureFactory_JointOpt"
OUTPUT_DIR = os.path.join(_BASE_OUT, TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FEATURE_WORKERS = max(1, (os.cpu_count() or 2))
print(f"[SYSTEM] Feature workers: {N_FEATURE_WORKERS}")

_perm_ans = input("Include Permutation scoring? (y/n, default=y): ").strip().lower()
USE_PERMUTATION_SCORING = (_perm_ans != 'n')
print(f"[CONFIG] Permutation scoring: {'ENABLED' if USE_PERMUTATION_SCORING else 'DISABLED'}\n")

TITAN_SYMBOLS = [
    'AA','AAL','AAPL','ABNB','ACWI','AEM','AFRM','AI','ALAB','ALB','AMAT','AMD','AMZN',
    'ANET','APA','APH','ARKK','AVGO','BA','BABA','BAC','BKR','BLDR','C','CARR','CAT',
    'CCJ','CCL','CE','CELH','CLF','CLSK','CMG','CNC','CPRT','CRM','CSCO','CSX','CVS',
    'CVX','DAL','DDOG','DHR','DIA','DIS','DKNG','DLTR','DOW','DVN','DXCM','EA','EBAY',
    'EEM','EMR','EQT','EWJ','EWT','EWW','EWY','EWZ','EXC','F','FANG','FCX','FITB',
    'FTNT','FTV','FXI','GBTC','GDX','GDXJ','GEHC','GFS','GIS','GOOG','GOOGL','GS',
    'HAL','HOOD','HPE','HPQ','HWM','IAU','IBM','IGV','IJH','IJR','INTC','IP','IR',
    'IWM','IYR','JNJ','KDP','KMI','KO','KRE','KWEB','LOW','LRCX','LUV','LVS','LYFT',
    'MAR','MARA','MCHP','MGM','MNST','MPC','MRK','MRNA','MRVL','MS','MSFT','MSTR',
    'MU','NCLH','NEE','NEM','NKE','NUE','NVDA','NVO','NXPI','ON','ORCL','OXY','PANW',
    'PCAR','PDD','PEP','PFE','PINS','PLTR','PYPL','QCOM','QQQ','QQQM','RBLX','RIOT',
    'RIVN','RTX','SBUX','SCHW','SHOP','SJM','SLB','SLV','SMCI','SMH','SNAP','SNOW',
    'SOFI','SOXX','SPLG','SPY','TER','TGT','TJX','TLT','TMUS','TQQQ','TSCO','TSLA',
    'TTD','TTWO','TWLO','TXN','U','UAL','UBER','UPS','USB','USO','VLO','VNQ','VRT',
    'VST','VT','VTR','WMT','WYNN','XBI','XLB','XLC','XLE','XLF','XLI','XLK','XLP',
    'XLRE','XLU','XLV','XLY','XOM','XOP','XRT',
]

# ==============================================================================
# MOMENTUM FEATURE CATALOG  v2.0
# (num, internal_key, display_description, pillar)
#
# 12 Anchors (A1-A12) + 5 Cross-Domain Ratios (R1-R5) = 17 seeds
# Pillars: Mismatch (7) | Regime (6) | Structure (4)
#
# Window scaling from indicator_n:
#   n_fast = max(2, n_s // 2)   where n_s = max(5, indicator_n // 2)
#   n_mid  = max(10, indicator_n)
#   n_slow = max(10, n_mid + n_s)
#   n_mem  = max(30, indicator_n * 3)
# ==============================================================================
MOMENTUM_FEATURE_CATALOG: list[tuple] = [
    ( 1, 'log_roc_fast',        'Log-space velocity  (n_fast log-return)',          'Mismatch'),
    ( 2, 'roc_fast',            'Arithmetic velocity  (n_fast pct-change)',         'Mismatch'),
    ( 3, 'geom_curv',           'Geometric Curvature  κ  (|y′′| / (1+y′²)^1.5)', 'Mismatch'),
    ( 4, 'rsi',                 'RSI  (Wilder, n_mid period)',                      'Mismatch'),
    ( 5, 'cpd_score',           'Changepoint Detection Score  ν',                   'Mismatch'),
    ( 6, 'cmo',                 'CMO  (Chande Momentum Oscillator)',                'Mismatch'),
    ( 7, 'lbr_pinball',         'LBR Pinball  (RSI of ROC_fast)',                  'Mismatch'),
    ( 8, 'psr',                 'Spectral Signature  PSR  (Welch low-freq ratio)',  'Regime'),
    ( 9, 'ctm',                 'Chande Trend Meter  (CTM, 4 BB periods)',          'Regime'),
    (10, 'octane',              'Octane Oscillator  (up/dn variance asymmetry)',    'Regime'),
    (11, 'dir_persist',         'Directional Persistence  (streak z-score)',        'Regime'),
    (12, 'rqa_det',             'RQA Determinism  DET v2  (diagonal runs ≥ 2)',    'Structure'),
    (13, 'curve_halflife',      'Curve / Half-Life ratio  (geometry × memory)',     'Structure'),
    (14, 'garr_er',             'GARR / ER  (log-concentration × path efficiency)','Structure'),
    (15, 'winding_persistence', 'Winding / Persistence  (topology × order)',       'Structure'),
    (16, 'pinball_garr',        'Pinball / GARR  (oscillator × struct momentum)',  'Mismatch'),
    (17, 'phase_dist_er',       'Phase Distance / ER  (2-D energy × efficiency)',  'Regime'),
]

_MCAT_BY_NUM = {n: (k, d, f) for n, k, d, f in MOMENTUM_FEATURE_CATALOG}
_MCAT_BY_KEY = {k: (n, d, f) for n, k, d, f in MOMENTUM_FEATURE_CATALOG}
ALL_MOMENTUM_KEYS = [k for _, k, _, _ in MOMENTUM_FEATURE_CATALOG]


def print_momentum_catalog() -> None:
    fams: dict[str, list] = {}
    for num, key, desc, fam in MOMENTUM_FEATURE_CATALOG:
        fams.setdefault(fam, []).append((num, desc))
    print("\n" + "═" * 72)
    print("  MOMENTUM FEATURE CATALOG  — pick the seeds to optimise this run")
    print("═" * 72)
    for fam, items in fams.items():
        print(f"\n  [{fam}]")
        for num, desc in items:
            print(f"    {num:>2}.  {desc}")
    print()
    print("  Syntax:  1,3,5-10,15   |   'all' = every feature")
    print("═" * 72)


def parse_selection(raw: str) -> list[str]:
    raw = raw.strip().lower()
    if raw == 'all':
        return list(ALL_MOMENTUM_KEYS)
    nums: set[int] = set()
    for tok in raw.split(','):
        tok = tok.strip()
        if '-' in tok:
            lo, hi = tok.split('-', 1)
            nums.update(range(int(lo), int(hi) + 1))
        elif tok.isdigit():
            nums.add(int(tok))
    return [_MCAT_BY_NUM[n][0] for n in sorted(nums) if n in _MCAT_BY_NUM]


# ==============================================================================
# BLOCK 2: NUMBA JIT ROLLING KERNELS
# ==============================================================================

@jit(nopython=True, cache=True)
def _lin_slope_nb(y):
    n = len(y)
    if n < 2: return 0.0
    xm = (n - 1) / 2.0
    ym = 0.0
    for i in range(n): ym += y[i]
    ym /= n
    num = den = 0.0
    for i in range(n):
        dx = i - xm
        num += dx * (y[i] - ym)
        den += dx * dx
    return num / den if den != 0.0 else 0.0

@jit(nopython=True, cache=True)
def _rolling_linslope(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _lin_slope_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _rolling_wma(arr, window):
    n = len(arr); out = np.full(n, np.nan)
    ws = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += arr[i - window + 1 + j] * (j + 1)
        out[i] = s / ws
    return out

@jit(nopython=True, cache=True)
def _kalman_numba(price, r=0.0001, q=0.001):
    xh = np.zeros_like(price); p = np.zeros_like(price)
    xh[0] = price[0]; p[0] = 1.0
    for t in range(1, len(price)):
        pm    = p[t-1] + q
        k     = pm / (pm + r)
        xh[t] = xh[t-1] + k * (price[t] - xh[t-1])
        p[t]  = (1 - k) * pm
    return xh

@jit(nopython=True, cache=True)
def _rsi_nb(prices, period=14):
    """Wilder's RSI — returns array same length as prices."""
    n = len(prices); rsi = np.full(n, 50.0)
    if n < period + 1: return rsi
    ag = al = 0.0
    for i in range(1, period + 1):
        d = prices[i] - prices[i - 1]
        if d > 0: ag += d
        else:     al += -d
    ag /= period; al /= period
    rsi[period] = 100.0 - 100.0 / (1.0 + ag / (al + 1e-9))
    for i in range(period + 1, n):
        d  = prices[i] - prices[i - 1]
        g  = d if d > 0 else 0.0
        l  = -d if d < 0 else 0.0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        rsi[i] = 100.0 - 100.0 / (1.0 + ag / (al + 1e-9))
    return rsi

@jit(nopython=True, cache=True)
def _rolling_geom_curv(velocity, acceleration):
    """κ = |y''| / (1 + y'²)^1.5  element-wise."""
    n = len(velocity); out = np.zeros(n)
    for i in range(n):
        vp = velocity[i]; ap = acceleration[i]
        out[i] = abs(ap) / ((1.0 + vp * vp) ** 1.5 + 1e-12)
    return out

@jit(nopython=True, cache=True)
def _winding_nb(x_z, y_z):
    """Winding number W = Σ Δθ / 2π  in (price_z, roc_z) phase space."""
    n = len(x_z)
    if n < 3: return 0.0
    total = 0.0; pi2 = 2.0 * 3.141592653589793
    for i in range(1, n):
        dt = np.arctan2(y_z[i], x_z[i]) - np.arctan2(y_z[i-1], x_z[i-1])
        while dt >  3.141592653589793: dt -= pi2
        while dt < -3.141592653589793: dt += pi2
        total += dt
    return total / pi2

@jit(nopython=True, cache=True)
def _rolling_phase_winding(price_z, roc_z, window):
    n = len(price_z); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _winding_nb(price_z[i-window+1:i+1], roc_z[i-window+1:i+1])
    return out

@jit(nopython=True, cache=True)
def _streak_zscore(returns, window):
    """Consecutive same-sign bars → z-score vs rolling history."""
    n = len(returns); streaks = np.zeros(n); out = np.zeros(n)
    s = 1
    for i in range(n):
        if i == 0: s = 1
        else:
            ps = 1 if returns[i-1] > 0 else (-1 if returns[i-1] < 0 else 0)
            cs = 1 if returns[i]   > 0 else (-1 if returns[i]   < 0 else 0)
            s  = s + 1 if (cs != 0 and cs == ps) else 1
        streaks[i] = float(s)
    for i in range(window - 1, n):
        w = streaks[i - window + 1: i + 1]
        mean = 0.0
        for v in w: mean += v
        mean /= window
        var = 0.0
        for v in w: var += (v - mean) ** 2
        std = (var / window) ** 0.5
        out[i] = (streaks[i] - mean) / (std + 1e-9)
    return out

# ── RQA Determinism v2 — diagonal lines of length ≥ 2 ────────────────────────
@jit(nopython=True, cache=True)
def _rqa_det_v2_nb(y, eps_factor=0.15):
    """DET = diagonal-line points (run ≥ 2) / total recurrent pairs (i≠j)."""
    n = len(y)
    if n < 5: return 0.0
    mean = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    var = 0.0
    for i in range(n): var += (y[i] - mean) ** 2
    eps = eps_factor * (var / n) ** 0.5
    if eps < 1e-10: return 0.0
    total = 0
    for i in range(n):
        for j in range(n):
            if i != j and abs(y[i] - y[j]) < eps:
                total += 1
    if total == 0: return 0.0
    diag_pts = 0
    for k in range(1, n):
        run_len = 0
        for i in range(n - k):
            if abs(y[i] - y[i + k]) < eps:
                run_len += 1
            else:
                if run_len >= 2: diag_pts += run_len
                run_len = 0
        if run_len >= 2: diag_pts += run_len
    return float(diag_pts) / float(total)

@jit(nopython=True, cache=True)
def _rolling_rqa_det_v2(arr, window, eps_factor=0.15):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _rqa_det_v2_nb(arr[i - window + 1: i + 1], eps_factor)
    return out


def _warm_up_numba():
    d  = np.random.randn(120).astype(np.float64)
    d2 = np.random.randn(120).astype(np.float64)
    _rolling_linslope(d, 10);  _rolling_wma(d, 10); _kalman_numba(d)
    _rsi_nb(d, 14)
    _rolling_geom_curv(d, d2); _rolling_phase_winding(d, d2, 20)
    _streak_zscore(d, 30);     _rolling_rqa_det_v2(d, 25)
    print("✅ Numba momentum kernels compiled and ready")

_warm_up_numba()


# ==============================================================================
# BLOCK 3: PARAMETERISED MOMENTUM FEATURE FACTORY  v2.0
#
# 17 seeds: 12 Anchors (A1-A12) + 5 Cross-Domain Ratios (R1-R5)
# Each selected seed → LENS_{z_n}_{name}_{z, z_slope, z_sos}
# Total LENS columns = len(selected_features) * 3  (no WIN columns)
#
# indicator_n  — primary lookback; scales all momentum windows:
#   n_s    = max(5,  indicator_n // 2)
#   n_fast = max(2,  n_s // 2)       ~3  at indicator_n=14
#   n_mid  = max(10, indicator_n)     =   indicator_n
#   n_slow = max(10, n_mid + n_s)    ~21 at indicator_n=14
#   n_mem  = max(30, indicator_n*3)  ~63 at indicator_n=21
#
# z_n          — single lens window; bounded seeds use rolling pct-rank
# BOUNDED      = {rsi, cmo, lbr_pinball, octane, dir_persist}
# ==============================================================================

def _psr_welch(roc_arr: np.ndarray, window: int, f_low_period: int) -> np.ndarray:
    """Power Spectral Ratio via Welch PSD. Falls back to numpy FFT if scipy absent."""
    n = len(roc_arr); out = np.full(n, np.nan)
    f_low = 1.0 / f_low_period
    for i in range(window - 1, n):
        seg = roc_arr[i - window + 1: i + 1].copy()
        seg -= seg.mean()
        if np.std(seg) < 1e-10:
            out[i] = 0.5; continue
        if _HAS_SCIPY:
            f, psd = _scipy_welch(seg, nperseg=len(seg), noverlap=0)
        else:
            psd = np.abs(np.fft.rfft(seg)) ** 2
            f   = np.fft.rfftfreq(len(seg))
        total  = psd.sum()
        out[i] = psd[f <= f_low].sum() / total if total > 1e-12 else 0.5
    return out


def generate_momentum_features_v2(df: pd.DataFrame,
                                   indicator_n: int = 20,
                                   z_n: int = 20,
                                   selected_features: list | None = None
                                   ) -> pd.DataFrame:
    """
    Generate 17-seed momentum features (12 anchors + 5 ratios).
    Each selected seed → z / z_slope / z_sos via single z_n lens.
    Total LENS columns = len(selected_features) * 3.
    """
    if selected_features is None:
        selected_features = ALL_MOMENTUM_KEYS
    sel = set(selected_features)

    df = df.copy()
    df['T_FINAL'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    cl  = df['close'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    idx = df.index
    n   = len(cl)

    # ── Window scaling ─────────────────────────────────────────────────────────
    n_s    = max(5,  indicator_n // 2)
    n_fast = max(2,  n_s // 2)          # ~3  at indicator_n=14
    n_mid  = max(10, indicator_n)        # = indicator_n
    n_slow = max(10, n_mid + n_s)        # ~21 at indicator_n=14
    n_mem  = max(30, indicator_n * 3)    # ~63 at indicator_n=21

    BOUNDED = {'rsi', 'cmo', 'lbr_pinball', 'octane', 'dir_persist'}

    cl_s = pd.Series(cl, index=idx)
    hi_s = pd.Series(hi, index=idx)
    lo_s = pd.Series(lo, index=idx)

    # ==========================================================================
    # ANCHORS (A1-A12)
    # ==========================================================================

    # A1. Log-space velocity
    log_roc_fast = np.log(cl_s / cl_s.shift(n_fast)).fillna(0).values

    # A2. Arithmetic velocity
    roc_fast = cl_s.pct_change(n_fast).fillna(0).values * 100

    # A3. Geometric curvature  κ = |a| / (1 + v²)^1.5
    velocity     = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)
    acceleration = np.diff(velocity, prepend=velocity[0])
    geom_curv    = _rolling_geom_curv(velocity, acceleration)

    # A4. Spectral Signature PSR  (Welch on 1-bar ROC stream, n_mem window)
    roc_1 = cl_s.pct_change(1).fillna(0).values.astype(np.float64)
    psr   = np.nan_to_num(_psr_welch(roc_1, window=n_mem, f_low_period=n_mem), nan=0.5)

    # A5. RQA Determinism v2  (diagonal runs ≥ 2, max(25, n_mem) window)
    rqa_det = _rolling_rqa_det_v2(roc_1, max(25, n_mem), 0.15)

    # A6. RSI — Wilder (n_mid period)
    delta    = cl_s.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / n_mid, min_periods=n_mid).mean()
    avg_loss = loss.ewm(alpha=1.0 / n_mid, min_periods=n_mid).mean()
    rsi      = (100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))).values

    # A7. Chande Trend Meter (CTM, 4 BB periods scaled to indicator_n)
    def _pct_b(series, period):
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        return ((series - (mid - 2*std)) / (4*std + 1e-9)) * 10

    ctm_p1 = max(20,  indicator_n)
    ctm_p2 = max(50,  indicator_n * 3)
    ctm_p3 = max(75,  indicator_n * 4)
    ctm_p4 = max(100, indicator_n * 5)
    ctm_raw = np.zeros(n)
    for p in [ctm_p1, ctm_p2, ctm_p3, ctm_p4]:
        for s in [cl_s, hi_s, lo_s]:
            ctm_raw += _pct_b(s, p).values
    ctm_raw += ((cl_s - cl_s.rolling(ctm_p4).mean()) /
                (cl_s.rolling(ctm_p4).std() + 1e-9) * 10).values
    ctm_raw += (cl_s.diff().gt(0).rolling(n_mid).mean() * 100 / 10).values
    chan_range = hi_s.rolling(ctm_p1).max() - lo_s.rolling(ctm_p1).min()
    ctm_raw   += ((cl_s - lo_s.rolling(ctm_p1).min()) /
                  (chan_range + 1e-9) * 10).values
    ctm = ctm_raw  # triple lens normalises; no min-max rescale needed

    # A8. Octane Oscillator — volatility asymmetry  [-1, +1]
    log_ret = np.log(cl_s / cl_s.shift(1)).fillna(0)
    up_var  = log_ret.where(log_ret > 0, 0.0).rolling(n_slow).var().values
    dn_var  = log_ret.where(log_ret < 0, 0.0).rolling(n_slow).var().values
    octane  = (up_var - dn_var) / (up_var + dn_var + 1e-10)

    # A9. Changepoint Detection Score  ν
    mu_short  = log_ret.rolling(n_fast).mean().values
    mu_long   = log_ret.rolling(n_slow).mean().values
    sig_long  = log_ret.rolling(n_slow).std().values
    cpd_score = (mu_short - mu_long) / (sig_long + 1e-10)

    # A10. Directional Persistence — streak z-score (n_mem window)
    daily_ret   = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)
    dir_persist = _streak_zscore(daily_ret, n_mem)

    # A11. CMO — Chande Momentum Oscillator  [-100, +100]
    delta_s = cl_s.diff().fillna(0)
    sum_up  = delta_s.where(delta_s > 0, 0.0).rolling(n_mid).sum().values
    sum_dn  = (-delta_s.where(delta_s < 0, 0.0)).rolling(n_mid).sum().values
    cmo     = ((sum_up - sum_dn) / (sum_up + sum_dn + 1e-10)) * 100

    # A12. LBR Pinball — RSI of ROC_fast  [0, 100]
    roc_fast_raw = cl_s.pct_change(n_fast).fillna(0) * 100
    d_pb  = roc_fast_raw.diff()
    g_pb  = d_pb.where(d_pb > 0, 0.0).rolling(n_fast).mean()
    l_pb  = (-d_pb.where(d_pb < 0, 0.0)).rolling(n_fast).mean()
    lbr_pinball = (100 - (100 / (1 + g_pb / (l_pb + 1e-9)))).values

    # ==========================================================================
    # SHARED INTERMEDIATES
    # ==========================================================================

    # GARR: log-return concentration  (n_fast rolling sum / n_slow rolling sum)
    garr = (log_ret.rolling(n_fast).sum() /
            (log_ret.rolling(n_slow).sum() + 1e-10)).values

    # Kaufman Efficiency Ratio (n_slow window)
    net_move  = (cl_s - cl_s.shift(n_slow)).abs()
    sum_steps = cl_s.diff().abs().rolling(n_slow).sum()
    er        = (net_move / (sum_steps + 1e-10)).values

    # Phase-space coordinates (normalised to n_slow rolling z)
    roc_fast_s    = pd.Series(roc_fast, index=idx)
    price_z_nslow = ((cl_s - cl_s.rolling(n_slow).mean()) /
                     (cl_s.rolling(n_slow).std() + 1e-10)).values.astype(np.float64)
    roc_z_nslow   = ((roc_fast_s - roc_fast_s.rolling(n_slow).mean()) /
                     (roc_fast_s.rolling(n_slow).std() + 1e-10)).values.astype(np.float64)

    # ==========================================================================
    # RATIOS (R1-R5)
    # ==========================================================================

    # R1. Curve / Half-Life
    roc_mid_arr     = cl_s.pct_change(n_mid).fillna(0).values * 100
    half_life_proxy = np.abs(roc_fast) / (np.abs(roc_mid_arr) + 1e-10)
    curve_halflife  = geom_curv / (half_life_proxy + 1e-10)

    # R2. GARR / ER
    garr_er = garr / (er + 1e-10)

    # R3. Winding / Persistence
    phase_wind          = _rolling_phase_winding(price_z_nslow, roc_z_nslow, window=n_slow)
    winding_persistence = phase_wind / (dir_persist + 1e-10)

    # R4. Pinball / GARR
    garr_scaled  = np.clip(garr * 100 + 50, 1.0, 100.0)
    pinball_garr = lbr_pinball / (garr_scaled + 1e-10)

    # R5. Phase Distance / ER
    dp            = np.diff(price_z_nslow, prepend=price_z_nslow[0])
    dr            = np.diff(roc_z_nslow,   prepend=roc_z_nslow[0])
    step_dist     = np.sqrt(dp**2 + dr**2)
    phase_dist    = pd.Series(step_dist, index=idx).rolling(n_slow).sum().values
    phase_dist_er = phase_dist / (er + 1e-10)

    # ==========================================================================
    # SEED DICT  (17 seeds: A1-A12 + R1-R5)
    # ==========================================================================
    ALL_SEEDS: dict = {
        'log_roc_fast':        pd.Series(log_roc_fast,        index=idx),
        'roc_fast':            pd.Series(roc_fast,            index=idx),
        'geom_curv':           pd.Series(geom_curv,           index=idx),
        'rsi':                 pd.Series(rsi,                 index=idx),
        'cpd_score':           pd.Series(cpd_score,           index=idx),
        'cmo':                 pd.Series(cmo,                 index=idx),
        'lbr_pinball':         pd.Series(lbr_pinball,         index=idx),
        'psr':                 pd.Series(psr,                 index=idx),
        'ctm':                 pd.Series(ctm,                 index=idx),
        'octane':              pd.Series(octane,              index=idx),
        'dir_persist':         pd.Series(dir_persist,         index=idx),
        'rqa_det':             pd.Series(rqa_det,             index=idx),
        'curve_halflife':      pd.Series(curve_halflife,      index=idx),
        'garr_er':             pd.Series(garr_er,             index=idx),
        'winding_persistence': pd.Series(winding_persistence, index=idx),
        'pinball_garr':        pd.Series(pinball_garr,        index=idx),
        'phase_dist_er':       pd.Series(phase_dist_er,       index=idx),
    }

    # ==========================================================================
    # Z-LENS TRANSFORM  (single z_n window)
    # z, z_slope, z_sos per selected seed → len(selected) * 3 LENS columns
    # Bounded seeds use rolling pct-rank as Z base (no z-score)
    # ==========================================================================
    for name, series in ALL_SEEDS.items():
        if name not in sel:
            continue
        arr   = series.values.astype(np.float64)
        arr_s = pd.Series(arr, index=idx)
        if name in BOUNDED:
            z = arr_s.rolling(z_n).rank(pct=True).values
        else:
            rm = arr_s.rolling(z_n).mean().values
            rs = arr_s.rolling(z_n).std().values
            z  = (arr - rm) / (rs + 1e-9)
        z_s     = pd.Series(z, index=idx)
        z_slope = z_s.diff().values
        z_sos   = pd.Series(z_slope, index=idx).diff().values
        df[f'LENS_{z_n}_{name}_z']       = z
        df[f'LENS_{z_n}_{name}_z_slope'] = z_slope
        df[f'LENS_{z_n}_{name}_z_sos']   = z_sos

    return (df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .dropna(subset=['T_FINAL'])
              .fillna(0))


# ==============================================================================
# BLOCK 4: PARALLEL DATA LOADER
# ==============================================================================
def fetch_data(symbol: str,
               start: str | None = None,
               end:   str | None = None) -> pd.DataFrame | None:
    # ── Primary: yf.download ──────────────────────────────────────────────────
    try:
        kw = dict(interval='1d', progress=False)
        if start and end:
            kw['start'] = str(start); kw['end'] = str(end)
        else:
            kw['period'] = '3y'
        data = yf.download(symbol, **kw)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]
            if 'close' in data.columns:
                return data
    except Exception:
        pass

    # ── Fallback: pandas_datareader / Stooq (works when Yahoo blocks Colab) ──
    try:
        from pandas_datareader import data as pdr
        if start and end:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
        else:
            e = pd.Timestamp.today()
            s = e - pd.DateOffset(years=3)
        data = pdr.get_data_stooq(symbol, start=s, end=e)
        if data is not None and not data.empty:
            data = data.sort_index()
            data.columns = [str(c).lower() for c in data.columns]
            if 'close' in data.columns:
                return data
    except Exception:
        pass

    return None


def load_hybrid_data_parallel(brain_name: str,
                               symbol_list: list,
                               indicator_n: int = 20,
                               z_n:         int = 20,
                               selected_features: list | None = None,
                               start_date = None,
                               end_date   = None,
                               dl_workers: int = 20) -> pd.DataFrame:
    if selected_features is None:
        selected_features = ALL_MOMENTUM_KEYS

    # Minimum bars: need enough for memory and PSR windows
    n_mem    = max(30, indicator_n * 3)
    min_bars = max(260, n_mem * 4 + 50)

    # Worker defined as LOCAL CLOSURE — no pickling, works with ThreadPoolExecutor
    def _worker(args):
        symbol, raw_dict = args
        try:
            raw_df = pd.DataFrame(raw_dict)
            raw_df.index = pd.to_datetime(raw_df.index)
            if len(raw_df) < min_bars:
                return None
            processed = generate_momentum_features_v2(
                raw_df, indicator_n, z_n, selected_features
            )
            if processed.empty:
                return None
            processed['symbol'] = symbol
            return processed.reset_index()
        except Exception:
            return None

    print(f"📥 Parallel download: {len(symbol_list)} symbols...")
    raw_results: dict = {}
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        fut_map = {pool.submit(fetch_data, sym, start_date, end_date): sym
                   for sym in symbol_list}
        for fut in tqdm(as_completed(fut_map), total=len(symbol_list),
                        desc="⬇ Downloading"):
            sym  = fut_map[fut]
            data = fut.result()
            if data is not None and len(data) >= min_bars:
                raw_results[sym] = data
    print(f"   ✅ {len(raw_results)}/{len(symbol_list)} symbols fetched")
    if not raw_results:
        return pd.DataFrame()

    work_items = [(sym, df.to_dict()) for sym, df in raw_results.items()]
    all_data: list = []
    print(f"⚙ Building momentum features (threads={N_FEATURE_WORKERS})...")
    with ThreadPoolExecutor(max_workers=N_FEATURE_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(work_items),
                        desc="⚙ Features"):
            try:
                result = fut.result()
                if result is not None:
                    all_data.append(result.set_index(result.columns[0]))
            except Exception as e:
                print(f"  ⚠️  Symbol failed: {e}")

    if not all_data:
        print("❌ No valid data after feature generation.")
        return pd.DataFrame()
    print(f"   ✅ {len(all_data)} symbols processed")
    return pd.concat(all_data, axis=0)


# ==============================================================================
# BLOCK 5: GPU-ACCELERATED AUDIT  (returns baseline_acc for the grid objective)
# ==============================================================================
def build_full_model(model_type, n_features, seq_len, device=DEVICE):
    with tf.device(device):
        model = Sequential([
            Input(shape=(seq_len, n_features)),
            GRU(128, return_sequences=True) if model_type == 'GRU'
                else LSTM(128, return_sequences=True),
            Dropout(0.2),
            GRU(64) if model_type == 'GRU' else LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1,  activation='sigmoid', dtype='float32'),
        ])
        model.compile(optimizer=Adam(1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model


@tf.function
def _eval_accuracy(model, X_b, y_b):
    preds = tf.squeeze(model(X_b, training=False), axis=-1)
    ok    = tf.equal(tf.cast(preds >= 0.5, tf.int32), tf.cast(y_b, tf.int32))
    return tf.reduce_mean(tf.cast(ok, tf.float32))


def run_judicial_audit(brain_name, master_df, model_type='GRU',
                       seq_len=10, epochs=30, batch_size=2048
                       ) -> tuple[pd.DataFrame, float]:
    """
    Returns
    -------
    report_df    : permutation-importance DataFrame  (Feature, I_raw)
    baseline_acc : scalar validation accuracy  ← used as grid objective
    """
    feat_cols  = [c for c in master_df.columns
                  if c.startswith('LENS_') or c.startswith('WIN_')]
    n_features = len(feat_cols)
    if n_features == 0:
        return pd.DataFrame(columns=['Feature', 'I_raw']), 0.0

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(master_df[feat_cols].values).astype(np.float32)
    y_raw    = master_df['T_FINAL'].values.astype(np.float32)

    n      = len(X_scaled)
    X_seqs = np.stack([X_scaled[i - seq_len:i] for i in range(seq_len, n)])
    y_seqs = y_raw[seq_len:]
    split  = int(len(X_seqs) * 0.8)
    X_tr, X_val = X_seqs[:split], X_seqs[split:]
    y_tr, y_val = y_seqs[:split], y_seqs[split:]
    print(f"  [DATA] train={len(X_tr):,}  val={len(X_val):,}  features={n_features}")

    AUTO  = tf.data.AUTOTUNE
    tr_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
             .shuffle(min(20_000, len(X_tr)), reshuffle_each_iteration=True)
             .batch(batch_size).prefetch(AUTO))
    va_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
             .batch(batch_size * 2).prefetch(AUTO))

    model = build_full_model(model_type, n_features, seq_len)
    with tf.device(DEVICE):
        model.fit(tr_ds, validation_data=va_ds, epochs=epochs,
                  callbacks=[
                      EarlyStopping(monitor='val_loss', patience=5,
                                    restore_best_weights=True),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=3, min_lr=1e-5),
                  ], verbose=1)

    Xvt = tf.constant(X_val)
    yvt = tf.constant(y_val)
    baseline_acc = float(_eval_accuracy(model, Xvt, yvt).numpy())
    print(f"  [MODEL] Baseline val accuracy: {baseline_acc:.4f}")

    rows = []
    if USE_PERMUTATION_SCORING:
        for fi, fname in enumerate(tqdm(feat_cols, desc="Permutation scoring")):
            try:
                Xp = X_val.copy()
                flat = Xp[:, :, fi].flatten()
                np.random.shuffle(flat)
                Xp[:, :, fi] = flat.reshape(Xp[:, :, fi].shape)
                pa = float(_eval_accuracy(model, tf.constant(Xp), yvt).numpy())
                rows.append({'Feature': fname, 'I_raw': max(0.0, baseline_acc - pa)})
            except Exception:
                rows.append({'Feature': fname, 'I_raw': 0.0})
    else:
        print("  [CONFIG] Permutation scoring skipped.")

    del model; gc.collect(); tf.keras.backend.clear_session()
    return pd.DataFrame(rows), baseline_acc


# ==============================================================================
# BLOCK 6: SOVEREIGN HUNT & DIVERSITY ANCHORS
# BRAIN_LOCKS start empty — auto-populated after the first full run.
# Momentum brain split:
#   DIRECTION → persistent directional momentum (CTM, Hurst-ROC, GARR, MHL)
#   EASE      → smooth clean signals (PSR, MTSI, dir_persist, LBR)
#   EXP       → explosive / exhaustion (accel, jerk, geom_curv, phase_wind)
# ==============================================================================
BRAIN_LOCKS: dict[str, list] = {'DIRECTION': [], 'EASE': [], 'EXP': []}


def _parse_feature_name(f):
    XFORM = {'z', 'slope', 'sos', 'pct'}
    if f.startswith('LENS_') or f.startswith('WIN_'):
        parts = f.split('_')
        prefix, window = parts[0], parts[1]
        rem = list(parts[2:])
        while rem and rem[-1] in XFORM:
            rem.pop()
        indicator = '_'.join(rem)
        family    = '_'.join(p for p in rem if not p.isdigit())
        return prefix, window, f'{prefix}_{window}_{indicator}', family
    return None, None, f, f


def apply_sovereign_hunt(ledger_df, master_data_df, brain_name, max_slots=19):
    locked  = BRAIN_LOCKS.get(brain_name, [])
    cands   = ledger_df.sort_values('I_raw', ascending=False)
    picked  = [f for f in locked if f in ledger_df['Feature'].values]
    for lf in locked:
        if lf not in ledger_df['Feature'].values:
            print(f"  ⚠️  BRAIN_LOCK '{lf}' not found")
    CORR_THR = 0.85
    fam_lbw: dict = {}
    for f in picked:
        _, _, lbw, fam = _parse_feature_name(f)
        fam_lbw.setdefault(fam, lbw)
    feat_cols   = [c for c in master_data_df.columns
                   if c.startswith('LENS_') or c.startswith('WIN_')]
    corr_matrix = master_data_df[feat_cols].corr()
    for _, row in cands.iterrows():
        if len(picked) >= max_slots: break
        fn = row['Feature']
        if fn in picked: continue
        _, _, fl, ff = _parse_feature_name(fn)
        if ff in fam_lbw and fam_lbw[ff] != fl: continue
        if picked and corr_matrix[fn].loc[picked].max() > CORR_THR: continue
        picked.append(fn)
        fam_lbw.setdefault(ff, fl)
    pca = PCA()
    pca.fit(RobustScaler().fit_transform(master_data_df[picked]))
    return picked, np.cumsum(pca.explained_variance_ratio_)


def generate_judicial_ledger(brain_name, report_df, master_data_df, iteration=1):
    df = report_df.copy()
    df['I_Norm'] = ((df['I_raw'] - df['I_raw'].min()) /
                    (df['I_raw'].max() - df['I_raw'].min() + 1e-9))
    picks, var_map = apply_sovereign_hunt(df, master_data_df, brain_name)
    csub  = master_data_df[picks].corr().abs()
    avg_c = ((csub.sum().sum() - len(picks)) /
             (len(picks)**2 - len(picks) + 1e-9))
    print(f"\n╔══ {brain_name} SOVEREIGN MOMENTUM CORE (Iter {iteration}) ══╗")
    print(f"║ {'RNK':<3} | {'MOMENTUM FEATURE':<35} | {'UV%':<4} | {'mR':<4} | {'IMPACT':<8} ║")
    print("╠" + "═"*4 + "╬" + "═"*37 + "╬" + "═"*6 + "╬" + "═"*6 + "╬" + "═"*10 + "╣")
    for i, fn in enumerate(picks):
        frow   = df[df['Feature'] == fn].iloc[0]
        locked = fn in BRAIN_LOCKS.get(brain_name, [])
        others = [p for p in picks if p != fn]
        max_r  = csub[fn].loc[others].max() if others else 0.0
        uv     = (1 - csub[fn].loc[others].mean()) * 100 if others else 100.0
        icon   = "🔒" if locked else "🔭"
        print(f"║ {i+1:02d}  | {icon} {fn[:33]:<33} | "
              f"{uv:>3.0f}% | {max_r:.2f} | {frow['I_Norm']:.4f} ║")
        df.loc[df['Feature'] == fn, ['UV%', 'Max_R', 'Is_Locked']] = [uv, max_r, locked]
    tv = var_map[-1] if len(var_map) > 0 else 0
    print("╠" + "═"*73 + "╣")
    print(f"║ PCA VARIANCE RETAINED:  {tv*100:>39.2f}% ║")
    print(f"║ AVG CROSS-CORRELATION:  {avg_c:>41.3f} ║")
    print(f"║ SLOTS FILLED:           {len(picks):>41}/19 ║")
    print("╚" + "═"*73 + "╝")
    return df[df['Feature'].isin(picks)]


# ==============================================================================
# GRID SEARCH OBJECTIVE
# ==============================================================================
def _grid_objective(ind_n: int, z_n: int,
                    symbols: list,
                    start_date, end_date,
                    selected_features: list,
                    brain_name: str,
                    model_type: str) -> float:
    """Returns validation accuracy (higher = better)."""
    master_df = load_hybrid_data_parallel(
        brain_name, symbols,
        indicator_n=ind_n, z_n=z_n,
        selected_features=selected_features,
        start_date=start_date, end_date=end_date,
    )
    if master_df.empty or len(master_df) < 500:
        return 0.0
    _, acc = run_judicial_audit(brain_name, master_df, model_type=model_type)
    print(f"  [EVAL] ind_n={ind_n:3d} | z_n={z_n:3d} | acc={acc:.4f}")
    return acc


# ==============================================================================
# BLOCK 7: COMMAND CENTER — Interactive Grid Search
# ==============================================================================
def run_lookback_tester():
    print("\n" + "═"*72)
    print("  Momentum Lookback × Z-Lens Optimizer")
    print("  Engine: Exhaustive grid over user-specified lookback lists")
    print("  Mandate: indicator_n and z_n optimised SIMULTANEOUSLY")
    print("═"*72)

    # ── Feature selection ──────────────────────────────────────────────────────
    print_momentum_catalog()
    raw_sel  = input("Select features to optimise this run: ").strip()
    selected = parse_selection(raw_sel)
    if not selected:
        print("⚠️  No valid features selected. Exiting."); return pd.DataFrame()
    print(f"\n  Selected {len(selected)} feature(s):")
    for key in selected:
        n, desc, fam = _MCAT_BY_KEY[key]
        print(f"    {n:>2}. [{fam:<14}] {desc}")

    # ── Lookback lists ─────────────────────────────────────────────────────────
    print()
    raw_ind = input("indicator_n values to test (e.g. 10,20,30,40,60): ").strip()
    raw_z   = input("z_n values to test         (e.g. 10,20,30,40,60): ").strip()

    def _parse_csv_ints(s: str) -> list[int]:
        out = []
        for tok in s.split(','):
            tok = tok.strip()
            if tok.isdigit():
                out.append(int(tok))
        return sorted(set(out))

    ind_list = _parse_csv_ints(raw_ind)
    z_list   = _parse_csv_ints(raw_z)
    if not ind_list or not z_list:
        print("⚠️  No valid values parsed. Exiting."); return pd.DataFrame()

    # ── Other inputs ───────────────────────────────────────────────────────────
    brain_ch   = input("Brain (1:DIRECTION / 2:EASE / 3:EXP) [default 1]: ").strip() or "1"
    brain_name = {'1': 'DIRECTION', '2': 'EASE', '3': 'EXP'}.get(brain_ch, 'DIRECTION')
    model_type = 'GRU' if brain_name == 'DIRECTION' else 'LSTM'
    num_syms   = int(input("Symbols per eval (e.g. 30): ") or "30")

    # ── Random 3-year window within the last 15 years ─────────────────────────
    cy  = _dt.now().year
    sy  = random.randint(cy - 15, cy - 3)
    sm  = random.randint(1, 12)
    start_date = datetime.date(sy, sm, 1)
    end_date   = start_date + datetime.timedelta(days=3 * 365)

    grid = list(itertools.product(ind_list, z_list))
    n_lens_cols = len(selected) * 3   # z, z_slope, z_sos per seed

    print(f"\n{'─'*72}")
    print(f"  Test window  : {start_date} → {end_date}  (3 years)")
    print(f"  Brain        : {brain_name}  ({model_type})")
    print(f"  indicator_n  : {ind_list}")
    print(f"  z_n          : {z_list}")
    print(f"  Combinations : {len(grid)}  ({len(ind_list)} × {len(z_list)})")
    print(f"  LENS cols    : {n_lens_cols} per (ind_n, z_n) pair  (no WIN columns)")
    print(f"{'─'*72}\n")

    # ── Sample symbols once (same pool for all evals) ─────────────────────────
    symbols = random.sample(TITAN_SYMBOLS, min(num_syms, len(TITAN_SYMBOLS)))

    # ── Exhaustive grid ────────────────────────────────────────────────────────
    best_acc, best_ind_n, best_z_n = -1.0, ind_list[0], z_list[0]
    results: list[tuple] = []

    for i, (ind_n, z_n) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] Testing ind_n={ind_n}  z_n={z_n} …")
        acc = _grid_objective(ind_n, z_n, symbols, start_date, end_date,
                              selected, brain_name, model_type)
        results.append((ind_n, z_n, acc))
        if acc > best_acc:
            best_acc, best_ind_n, best_z_n = acc, ind_n, z_n

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  GRID SEARCH — FULL RESULTS")
    print(f"  {'#':>3}  {'ind_n':>6}  {'z_n':>5}  {'acc':>8}")
    print(f"  {'─'*35}")
    for i, (ind_n, z_n, acc) in enumerate(results, 1):
        marker = " ← best" if (ind_n == best_ind_n and z_n == best_z_n) else ""
        print(f"  {i:>3}  {ind_n:>6}  {z_n:>5}  {acc:>8.4f}{marker}")

    # ── Final audit at optimal pair ────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  🏆 OPTIMAL PAIR: indicator_n={best_ind_n}  z_n={best_z_n}")
    print(f"     Validation accuracy = {best_acc:.4f}")
    print(f"{'═'*72}")
    print("\n  Running final full audit at optimal pair …\n")

    master_df = load_hybrid_data_parallel(
        brain_name, symbols,
        indicator_n=best_ind_n, z_n=best_z_n,
        selected_features=selected,
        start_date=start_date, end_date=end_date,
    )
    if master_df.empty:
        print("⚠️  No data for final audit."); return pd.DataFrame()

    report_raw, _ = run_judicial_audit(brain_name, master_df,
                                        model_type=model_type)
    ledger = generate_judicial_ledger(brain_name, report_raw,
                                       master_df, iteration='FINAL')

    # ── Export ─────────────────────────────────────────────────────────────────
    fname = (f"MomentumOpt_{brain_name}_"
             f"ind{best_ind_n}_z{best_z_n}_"
             f"{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv")
    fpath = os.path.join(OUTPUT_DIR, fname)
    ledger.to_csv(fpath, index=False)
    print(f"\n📂 Final ledger saved → {fpath}")

    # ── Sync winning pair to BRAIN_LOCKS ──────────────────────────────────────
    if not ledger.empty and 'Feature' in ledger.columns:
        BRAIN_LOCKS[brain_name] = ledger['Feature'].tolist()
        print(f"✅ BRAIN_LOCKS['{brain_name}'] updated with "
              f"{len(BRAIN_LOCKS[brain_name])} features")

    return ledger


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_lookback_tester()
