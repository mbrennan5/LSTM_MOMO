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
# MOMENTUM FEATURE CATALOG
# (num, internal_key, display_description, family)
#
# Window notation in descriptions uses indicator_n-derived names:
#   n_fast  ≈ indicator_n // 4    n_mid  ≈ indicator_n // 2
#   n_slow  ≈ indicator_n         n_long ≈ indicator_n × 2
#   n_mem   ≈ indicator_n × 3     (MHL / Hurst memory window)
# ==============================================================================
MOMENTUM_FEATURE_CATALOG: list[tuple] = [
    ( 1, 'ctm',         'Chande Trend Meter  (CTM, 4 BB periods)',         'Trend-Strength'),
    ( 2, 'rsi',         'RSI  (Wilder, n_fast period)',                    'Oscillator'),
    ( 3, 'roc_fast',    'ROC Fast   (n_fast bars)',                        'Momentum'),
    ( 4, 'roc_mid',     'ROC Mid    (n_mid bars)',                         'Momentum'),
    ( 5, 'roc_slow',    'ROC Slow   (n_slow bars)',                        'Momentum'),
    ( 6, 'roc_long',    'ROC Long   (n_long bars)',                        'Momentum'),
    ( 7, 'roc_vlong',   'ROC VLong  (n_vlong = 2×n_long bars)',            'Momentum'),
    ( 8, 'roc_fast_log','Log-ROC Fast  (n_fast log-return)',               'Momentum'),
    ( 9, 'roc_mid_log', 'Log-ROC Mid   (n_mid log-return)',                'Momentum'),
    (10, 'roc_long_log','Log-ROC Long  (n_long log-return)',               'Momentum'),
    (11, 'mhl',         'Autocorrelation Half-Life  (n_mem window)',       'Memory'),
    (12, 'hurst_roc',   'Hurst Exponent on ROC R/S  (n_mem window)',       'Memory'),
    (13, 'garr_ratio',  'GARR Ratio  (n_slow geom / n_annual geom)',       'Momentum'),
    (14, 'accel_fs',    'ROC Acceleration  fast − slow',                   'Acceleration'),
    (15, 'accel_fm',    'ROC Acceleration  fast − mid',                    'Acceleration'),
    (16, 'accel_ms',    'ROC Acceleration  mid  − slow',                   'Acceleration'),
    (17, 'jerk',        'Momentum Jerk  (Δ accel_fs)',                     'Acceleration'),
    (18, 'jerk_med',    'Momentum Jerk Medium  (Δ accel_fm)',              'Acceleration'),
    (19, 'geom_curv',   'Geometric Curvature  κ  (|y′′| / (1+y′²)^1.5)', 'Curvature'),
    (20, 'quad_gamma',  'Quadratic Curvature  γ  (t² regression coeff)',  'Curvature'),
    (21, 'lbr_pinball', 'LBR Pinball  (RSI_fast of ROC_fast)',             'Oscillator'),
    (22, 'phase_wind',  'Phase-Space Winding  W  (price_z × ROC_z)',      'Oscillator'),
    (23, 'psr',         'Spectral Signature  PSR  (low-freq power ratio)', 'Spectral'),
    (24, 'mtsi',        'Modified True Strength Index  MTSI-v2',           'Oscillator'),
    (25, 'dir_persist', 'Directional Persistence  (streak z-score)',       'Memory'),
    (26, 'rqa_det',     'RQA Determinism  DET  (diagonal recurrence)',     'Structural'),
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
# (trend originals kept for compatibility + all momentum-specific additions)
# ==============================================================================

# ── Shared with Trend Tester ───────────────────────────────────────────────────
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

# ── Momentum-specific additions ────────────────────────────────────────────────
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
def _hurst_rs_nb(y):
    """Rescaled-range Hurst exponent.  H > 0.5 = persistent momentum."""
    n = len(y)
    if n < 8: return 0.5
    mean = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    cum = cmin = cmax = 0.0
    for i in range(n):
        cum += y[i] - mean
        if cum < cmin: cmin = cum
        if cum > cmax: cmax = cum
    R = cmax - cmin
    var = 0.0
    for i in range(n): var += (y[i] - mean) ** 2
    S = (var / n) ** 0.5
    if S < 1e-12 or R <= 0.0: return 0.5
    h = np.log(R / S) / np.log(float(n))
    if h < 0.0 or h > 1.0 or np.isnan(h): return 0.5
    return h

@jit(nopython=True, cache=True)
def _rolling_hurst_rs(arr, window):
    n = len(arr); out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _hurst_rs_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _mhl_nb(y, max_lag=30):
    """Smallest lag τ where ACF(τ) ≤ 1/e ≈ 0.3679."""
    n = len(y)
    if n < 10: return float(max_lag)
    mean = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    var = 0.0
    for i in range(n): var += (y[i] - mean) ** 2
    if var < 1e-10: return float(max_lag)
    threshold = 1.0 / 2.718281828
    for lag in range(1, min(max_lag, n // 2) + 1):
        cov = 0.0
        for i in range(lag, n):
            cov += (y[i] - mean) * (y[i - lag] - mean)
        if cov / var <= threshold:
            return float(lag)
    return float(max_lag)

@jit(nopython=True, cache=True)
def _rolling_mhl(arr, window, max_lag=30):
    n = len(arr); out = np.full(n, float(max_lag))
    for i in range(window - 1, n):
        out[i] = _mhl_nb(arr[i - window + 1: i + 1], max_lag)
    return out

@jit(nopython=True, cache=True)
def _rqa_det_nb(y, eps_factor=0.15):
    """DET = diagonal-line recurrent points / total recurrent points."""
    n = len(y)
    if n < 5: return 0.5
    mean = var = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    for i in range(n): var += (y[i] - mean) ** 2
    eps = eps_factor * (var / n) ** 0.5
    if eps < 1e-10: return 0.5
    total = diag = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y[i] - y[j]) < eps:
                total += 1
                if i > 0 and j > 0 and abs(y[i-1] - y[j-1]) < eps:
                    diag += 1
    return diag / total if total > 0 else 0.0

@jit(nopython=True, cache=True)
def _rolling_rqa_det(arr, window):
    n = len(arr); out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _rqa_det_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _rolling_geom_curv(velocity, acceleration):
    """κ = |y''| / (1 + y'²)^1.5  element-wise."""
    n = len(velocity); out = np.zeros(n)
    for i in range(n):
        vp = velocity[i]; ap = acceleration[i]
        out[i] = abs(ap) / ((1.0 + vp * vp) ** 1.5 + 1e-12)
    return out

@jit(nopython=True, cache=True)
def _quad_gamma_nb(y):
    """OLS coefficient of t² when regressing y on [1, t, t²]."""
    n = len(y)
    if n < 6: return 0.0
    s1=st=st2=st3=st4=sy=sty=st2y=0.0
    for i in range(n):
        t=float(i); t2=t*t
        s1+=1.0; st+=t; st2+=t2; st3+=t2*t; st4+=t2*t2
        sy+=y[i]; sty+=t*y[i]; st2y+=t2*y[i]
    A = np.zeros((3, 4))
    A[0,0]=s1;  A[0,1]=st;  A[0,2]=st2; A[0,3]=sy
    A[1,0]=st;  A[1,1]=st2; A[1,2]=st3; A[1,3]=sty
    A[2,0]=st2; A[2,1]=st3; A[2,2]=st4; A[2,3]=st2y
    for col in range(3):
        max_val=abs(A[col,col]); max_row=col
        for row in range(col+1, 3):
            if abs(A[row,col]) > max_val:
                max_val=abs(A[row,col]); max_row=row
        if max_row != col:
            for k in range(4):
                tmp=A[col,k]; A[col,k]=A[max_row,k]; A[max_row,k]=tmp
        if abs(A[col,col]) < 1e-12: return 0.0
        for row in range(col+1, 3):
            f=A[row,col]/A[col,col]
            for k in range(col, 4):
                A[row,k]-=f*A[col,k]
    x=np.zeros(3)
    for i in range(2,-1,-1):
        x[i]=A[i,3]
        for j in range(i+1, 3):
            x[i]-=A[i,j]*x[j]
        x[i]/=A[i,i] if abs(A[i,i]) > 1e-12 else 1.0
    return x[2]

@jit(nopython=True, cache=True)
def _rolling_quad_gamma(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _quad_gamma_nb(arr[i - window + 1: i + 1])
    return out

@jit(nopython=True, cache=True)
def _winding_nb(x_z, y_z):
    """Winding number W = Σ Δθ / 2π  in (price_z, roc_z) phase space."""
    n=len(x_z)
    if n < 3: return 0.0
    total=0.0; pi2=2.0*3.141592653589793
    for i in range(1, n):
        dt=np.arctan2(y_z[i],x_z[i])-np.arctan2(y_z[i-1],x_z[i-1])
        while dt >  3.141592653589793: dt -= pi2
        while dt < -3.141592653589793: dt += pi2
        total += dt
    return total/pi2

@jit(nopython=True, cache=True)
def _rolling_phase_winding(price_z, roc_z, window):
    n=len(price_z); out=np.full(n,0.0)
    for i in range(window-1, n):
        out[i]=_winding_nb(price_z[i-window+1:i+1], roc_z[i-window+1:i+1])
    return out

@jit(nopython=True, cache=True)
def _streak_zscore(returns, window):
    """Consecutive same-sign bars → z-score vs rolling history."""
    n=len(returns); streaks=np.zeros(n); out=np.zeros(n)
    s=1
    for i in range(n):
        if i==0: s=1
        else:
            ps=1 if returns[i-1]>0 else (-1 if returns[i-1]<0 else 0)
            cs=1 if returns[i]  >0 else (-1 if returns[i]  <0 else 0)
            s = s+1 if (cs!=0 and cs==ps) else 1
        streaks[i]=float(s)
    for i in range(window-1, n):
        w=streaks[i-window+1:i+1]
        mean=0.0
        for v in w: mean+=v
        mean/=window
        var=0.0
        for v in w: var+=(v-mean)**2
        std=(var/window)**0.5
        out[i]=(streaks[i]-mean)/(std+1e-9)
    return out


def _warm_up_numba():
    d  = np.random.randn(120).astype(np.float64)
    d2 = np.random.randn(120).astype(np.float64)
    _rolling_linslope(d, 10);  _rolling_wma(d, 10); _kalman_numba(d)
    _rsi_nb(d, 14);            _rolling_hurst_rs(d, 40)
    _rolling_mhl(d, 63);       _rolling_rqa_det(d, 40)
    _rolling_geom_curv(d, d2); _rolling_quad_gamma(d, 30)
    _rolling_phase_winding(d, d2, 20); _streak_zscore(d, 30)
    print("✅ Numba momentum kernels compiled and ready")

_warm_up_numba()


# ==============================================================================
# BLOCK 3: PARAMETERISED MOMENTUM FEATURE FACTORY
#
# indicator_n  — primary lookback; all momentum windows scale from this:
#     n_s     = max(5,   indicator_n // 2)   short
#     n_m     = max(10,  indicator_n)         medium / primary
#     n_l     = max(20,  indicator_n * 2)     long
#     n_xl    = max(30,  indicator_n * 3)     extra-long / memory
#
#     Derived windows (all clamped to sensible minimums):
#       n_roc_fast  = max(2,   n_s // 2)       ~2 bars
#       n_roc_mid   = max(3,   n_s)             ~5 bars
#       n_roc_slow  = max(7,   n_m // 2)        ~7-10 bars
#       n_roc_long  = max(21,  n_m)             ~20 bars
#       n_roc_vlong = max(30,  n_l)             ~40 bars
#       n_rsi       = max(3,   n_s)             RSI period
#       n_mem       = max(30,  n_xl)            MHL / Hurst window
#       n_garr_s    = max(21,  n_m)             GARR short (≈1 month)
#       n_garr_l    = max(252, n_m * 12)        GARR long  (≈1 year)
#       n_accel_f   = max(7,   n_m // 3)        Accel fast ROC
#       n_accel_m   = max(14,  n_m // 2)        Accel mid  ROC
#       n_accel_s   = max(21,  n_m)             Accel slow ROC
#       n_quad      = max(60,  n_xl * 2)        Quadratic gamma window
#       n_rqa       = max(20,  n_l)             RQA Determinism window
#       n_psr       = max(63,  n_l * 2)         PSR spectral window
#       n_streak    = max(30,  n_xl)            Directional persistence
#       n_phase     = max(10,  n_s)             Phase-space winding
#
# z_n          — the SINGLE lens window (replaces fixed LENS_10 / LENS_90).
#                Each selected feature gets z, z_slope, z_sos columns.
#
# selected_features — list of keys from MOMENTUM_FEATURE_CATALOG.
#                     All intermediates are computed; only selected seeds
#                     enter the Z-lens pipeline.
# ==============================================================================

def _psr_numpy(roc_arr, window=126, f_low_period=63):
    """Power Spectral Ratio: fraction of power at f ≤ 1/f_low_period."""
    n = len(roc_arr); out = np.full(n, np.nan)
    f_low = 1.0 / f_low_period
    for i in range(window - 1, n):
        seg  = roc_arr[i - window + 1: i + 1]
        seg  = seg - seg.mean()
        psd  = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(window)
        total = psd.sum()
        out[i] = psd[freqs <= f_low].sum() / total if total > 1e-12 else 0.5
    return out


def generate_momentum_features_v2(df: pd.DataFrame,
                                   indicator_n: int = 20,
                                   z_n: int = 20,
                                   selected_features: list | None = None
                                   ) -> pd.DataFrame:
    if selected_features is None:
        selected_features = ALL_MOMENTUM_KEYS
    sel = set(selected_features)

    df = df.copy()
    df['hlc3']    = (df['high'] + df['low'] + df['close']) / 3
    df['T_FINAL'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    cl  = df['close'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    idx = df.index
    n   = len(cl)

    # ── Window family ──────────────────────────────────────────────────────────
    n_s  = max(5,  indicator_n // 2)
    n_m  = max(10, indicator_n)
    n_l  = max(20, indicator_n * 2)
    n_xl = max(30, indicator_n * 3)

    n_roc_fast  = max(2,  n_s // 2)
    n_roc_mid   = max(3,  n_s)
    n_roc_slow  = max(7,  n_m // 2)
    n_roc_long  = max(21, n_m)
    n_roc_vlong = max(30, n_l)
    n_rsi       = max(3,  n_s)
    n_mem       = max(30, n_xl)
    n_garr_s    = max(21, n_m)
    n_garr_l    = max(252, n_m * 12)
    n_accel_f   = max(7,  n_m // 3)
    n_accel_m   = max(14, n_m // 2)
    n_accel_s   = max(21, n_m)
    n_quad      = max(60, n_xl * 2)
    n_rqa       = max(20, n_l)
    n_psr       = max(63, n_l * 2)
    n_streak    = max(30, n_xl)
    n_phase     = max(10, n_s)

    cl_s  = pd.Series(cl,  index=idx)
    hi_s  = pd.Series(hi,  index=idx)
    lo_s  = pd.Series(lo,  index=idx)
    vol_s = pd.Series(vol, index=idx)

    # ── 1. CTM — Chande Trend Meter ────────────────────────────────────────────
    ctm_p1 = max(20,  indicator_n)
    ctm_p2 = max(50,  indicator_n * 3)
    ctm_p3 = max(75,  indicator_n * 4)
    ctm_p4 = max(100, indicator_n * 5)

    ctm_raw = np.zeros(n)
    for period in [ctm_p1, ctm_p2, ctm_p3, ctm_p4]:
        for arr_s, arr_v in [(cl_s, cl), (hi_s, hi), (lo_s, lo)]:
            sma  = arr_s.rolling(period).mean().values
            std  = arr_s.rolling(period).std().values
            upper = sma + 2 * std; lower = sma - 2 * std
            ctm_raw += (arr_v - lower) / (upper - lower + 1e-9) * 10

    ctm_base = max(ctm_p4, 100)
    sma_base = cl_s.rolling(ctm_base).mean().values
    std_base = cl_s.rolling(ctm_base).std().values
    ctm_raw += (cl - sma_base) / (std_base + 1e-9) * 10

    rsi_ctm = _rsi_nb(cl, n_rsi)
    ctm_raw += rsi_ctm / 10.0

    lo2 = lo_s.rolling(2).min().values
    hi2 = hi_s.rolling(2).max().values
    ctm_raw += (cl - lo2) / (hi2 - lo2 + 1e-9) * 10

    ctm_s    = pd.Series(ctm_raw, index=idx)
    roll_min = ctm_s.rolling(min(252, n), min_periods=1).min().values
    roll_max = ctm_s.rolling(min(252, n), min_periods=1).max().values
    ctm      = (ctm_raw - roll_min) / (roll_max - roll_min + 1e-9) * 100

    # ── 2. RSI ─────────────────────────────────────────────────────────────────
    rsi_arr = _rsi_nb(cl, n_rsi)

    # ── ROC family ─────────────────────────────────────────────────────────────
    def _roc(p): return cl_s.pct_change(p).fillna(0).values * 100

    roc_fast  = _roc(n_roc_fast)
    roc_mid   = _roc(n_roc_mid)
    roc_slow  = _roc(n_roc_slow)
    roc_long  = _roc(n_roc_long)
    roc_vlong = _roc(n_roc_vlong)

    log_cl       = np.log(cl + 1e-9)
    def _log_roc(p):
        return np.diff(log_cl, n=p, prepend=[log_cl[0]] * p)

    roc_fast_log  = _log_roc(n_roc_fast)
    roc_mid_log   = _log_roc(n_roc_mid)
    roc_long_log  = _log_roc(n_roc_long)

    # ── 3. MHL ─────────────────────────────────────────────────────────────────
    mhl = _rolling_mhl(roc_mid, window=n_mem, max_lag=max(15, n_mem // 3))

    # ── 4. Hurst on ROC ────────────────────────────────────────────────────────
    hurst_roc = _rolling_hurst_rs(roc_mid, window=n_mem)

    # ── 5. GARR ratio ──────────────────────────────────────────────────────────
    log_ret   = np.log(cl_s / cl_s.shift(1)).fillna(0)
    garr_s    = log_ret.rolling(n_garr_s).mean().apply(np.exp) - 1
    garr_l    = log_ret.rolling(n_garr_l).mean().apply(np.exp) - 1
    garr_ratio = (garr_s / (garr_l.abs() + 1e-9) *
                  garr_l.apply(np.sign)).values

    # ── 6. ROC Acceleration & Jerk ─────────────────────────────────────────────
    roc_acf = cl_s.pct_change(n_accel_f).fillna(0).values * 100
    roc_acm = cl_s.pct_change(n_accel_m).fillna(0).values * 100
    roc_acs = cl_s.pct_change(n_accel_s).fillna(0).values * 100

    accel_fs  = roc_acf - roc_acs
    accel_fm  = roc_acf - roc_acm
    accel_ms  = roc_acm - roc_acs
    jerk      = np.diff(accel_fs, prepend=accel_fs[0])
    jerk_med  = np.diff(accel_fm, prepend=accel_fm[0])

    # ── 7. Geometric Curvature κ ───────────────────────────────────────────────
    velocity     = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)
    acceleration = np.diff(velocity, prepend=velocity[0])
    geom_curv    = _rolling_geom_curv(velocity, acceleration)

    # ── 8. Quadratic Curvature γ ───────────────────────────────────────────────
    daily_ret  = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)
    quad_gamma = _rolling_quad_gamma(daily_ret, window=n_quad)

    # ── 9. LBR Pinball — RSI_fast of ROC_fast ─────────────────────────────────
    lbr_roc_input = cl_s.pct_change(n_roc_fast).fillna(0).values * 100
    lbr_pinball   = _rsi_nb(lbr_roc_input, period=max(3, n_rsi // 2))

    # ── 10. Phase-Space Winding ────────────────────────────────────────────────
    def _zscore_roll(arr, w):
        s = pd.Series(arr, index=idx)
        rm = s.rolling(w).mean().values
        rs = s.rolling(w).std().values
        return (arr - rm) / (rs + 1e-9)

    price_z = _zscore_roll(cl,        n_phase).astype(np.float64)
    roc_z   = _zscore_roll(roc_mid,   n_phase).astype(np.float64)
    phase_wind = _rolling_phase_winding(price_z, roc_z, window=n_phase)

    # ── 11. PSR — Power Spectral Ratio ─────────────────────────────────────────
    psr = _psr_numpy(roc_mid, window=n_psr, f_low_period=max(10, n_psr // 2))
    psr = np.nan_to_num(psr, nan=0.5)

    # ── 12. MTSI-v2 ────────────────────────────────────────────────────────────
    tp_v         = pd.Series(cl * vol, index=idx)
    vwap_2       = (tp_v.rolling(2).sum() / (vol_s.rolling(2).sum() + 1e-9)).values
    diff_mtsi    = np.log(cl / (np.abs(vwap_2) + 1e-9))
    diff_s       = pd.Series(diff_mtsi, index=idx)
    abs_diff_s   = pd.Series(np.abs(diff_mtsi), index=idx)
    dbl_diff     = diff_s.ewm(span=3, adjust=False).mean().ewm(span=2, adjust=False).mean()
    dbl_abs_diff = abs_diff_s.ewm(span=3, adjust=False).mean().ewm(span=2, adjust=False).mean()
    mtsi         = (100.0 * dbl_diff / (dbl_abs_diff + 1e-9)).values

    # ── 13. Directional Persistence ────────────────────────────────────────────
    dir_persist = _streak_zscore(daily_ret, window=n_streak)

    # ── 14. RQA Determinism ────────────────────────────────────────────────────
    rqa_det = _rolling_rqa_det(roc_mid, window=n_rqa)

    # ── Seed dictionary ────────────────────────────────────────────────────────
    SEEDS: dict[str, pd.Series] = {
        'ctm':          pd.Series(ctm,          index=idx),
        'rsi':          pd.Series(rsi_arr,       index=idx),
        'roc_fast':     pd.Series(roc_fast,      index=idx),
        'roc_mid':      pd.Series(roc_mid,       index=idx),
        'roc_slow':     pd.Series(roc_slow,      index=idx),
        'roc_long':     pd.Series(roc_long,      index=idx),
        'roc_vlong':    pd.Series(roc_vlong,     index=idx),
        'roc_fast_log': pd.Series(roc_fast_log,  index=idx),
        'roc_mid_log':  pd.Series(roc_mid_log,   index=idx),
        'roc_long_log': pd.Series(roc_long_log,  index=idx),
        'mhl':          pd.Series(mhl,           index=idx),
        'hurst_roc':    pd.Series(hurst_roc,     index=idx),
        'garr_ratio':   pd.Series(garr_ratio,    index=idx),
        'accel_fs':     pd.Series(accel_fs,      index=idx),
        'accel_fm':     pd.Series(accel_fm,      index=idx),
        'accel_ms':     pd.Series(accel_ms,      index=idx),
        'jerk':         pd.Series(jerk,          index=idx),
        'jerk_med':     pd.Series(jerk_med,      index=idx),
        'geom_curv':    pd.Series(geom_curv,     index=idx),
        'quad_gamma':   pd.Series(quad_gamma,    index=idx),
        'lbr_pinball':  pd.Series(lbr_pinball,   index=idx),
        'phase_wind':   pd.Series(phase_wind,    index=idx),
        'psr':          pd.Series(psr,           index=idx),
        'mtsi':         pd.Series(mtsi,          index=idx),
        'dir_persist':  pd.Series(dir_persist,   index=idx),
        'rqa_det':      pd.Series(rqa_det,       index=idx),
    }

    # ── Apply Z-lens (Physics Trio) to SELECTED seeds only ────────────────────
    # LENS_{z_n}_{key}_z          → Position   (z-score vs rolling mean/std)
    # LENS_{z_n}_{key}_z_slope    → Velocity   (rate of change of z)
    # LENS_{z_n}_{key}_z_sos      → Acceleration / SOS  (turning-point signal)
    for name, series in SEEDS.items():
        if name not in sel:
            continue
        arr = series.values.astype(np.float64)
        rm  = pd.Series(arr, index=idx).rolling(z_n).mean().values
        rs  = pd.Series(arr, index=idx).rolling(z_n).std().values
        z   = (arr - rm) / (rs + 1e-9)
        zs  = _rolling_linslope(z,  z_n)
        zso = _rolling_linslope(zs, z_n)
        df[f'LENS_{z_n}_{name}_z']       = z
        df[f'LENS_{z_n}_{name}_z_slope'] = zs
        df[f'LENS_{z_n}_{name}_z_sos']   = zso

    # ── WIN rolling-pct for unbounded / structural seeds (when selected) ───────
    WIN_SEEDS = {'mhl', 'hurst_roc', 'garr_ratio', 'accel_fs',
                 'jerk', 'geom_curv', 'quad_gamma', 'phase_wind', 'rqa_det'}
    win_a = max(5,  z_n // 3)
    win_b = z_n
    for name in WIN_SEEDS:
        if name not in sel:
            continue
        arr = SEEDS[name].values
        for win in [win_a, win_b]:
            rm = pd.Series(arr, index=idx).rolling(win).mean().values
            df[f'WIN_{win}_{name}_pct'] = arr / (np.abs(rm) + 1e-9) - 1.0

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
    try:
        kw = dict(interval='1d', progress=False)
        if start and end:
            kw['start'] = str(start); kw['end'] = str(end)
        else:
            kw['period'] = '3y'
        data = yf.download(symbol, **kw)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
        return data if 'close' in data.columns else None
    except Exception:
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

    # Minimum bars: need enough for GARR_long and memory windows
    n_m      = max(10, indicator_n)
    n_garr_l = max(252, n_m * 12)
    n_xl     = max(30, indicator_n * 3)
    n_psr    = max(63, max(20, indicator_n * 2) * 2)
    min_bars = max(260, n_garr_l + 50, n_psr + 50)

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
    n_win_seeds = len({'mhl','hurst_roc','garr_ratio','accel_fs',
                       'jerk','geom_curv','quad_gamma','phase_wind','rqa_det'}
                      & set(selected))

    print(f"\n{'─'*72}")
    print(f"  Test window  : {start_date} → {end_date}  (3 years)")
    print(f"  Brain        : {brain_name}  ({model_type})")
    print(f"  indicator_n  : {ind_list}")
    print(f"  z_n          : {z_list}")
    print(f"  Combinations : {len(grid)}  ({len(ind_list)} × {len(z_list)})")
    print(f"  LENS cols    : {n_lens_cols} LENS + {n_win_seeds*2} WIN per pair")
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
