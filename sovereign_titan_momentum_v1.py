# ==============================================================================
# SOVEREIGN TITAN v1.0 — MOMENTUM EDITION  (Ron Harper Adaptation)
# Converts the v4.30 Trend Tester framework into a Momentum Tester.
#
# 14 Momentum Measures implemented in the Feature Factory:
#   1.  Chande Trend Meter          (CTM)
#   2.  Raw ROC / Price Momentum    (n = 2, 3, 30) + log variants
#   3.  Autocorrelation Half-Life   (MHL)
#   4.  Hurst Exponent on ROC       (R/S method)
#   5.  GARR Ratio                  (Geometric Avg Rate of Return: 1-mo / 12-mo)
#   6.  ROC Acceleration & Jerk     (fast=7, medium=14, slow=21)
#   7.  Geometric Curvature         (κ — differential formula)
#   8.  Quadratic Curvature         (γ — t² regression coefficient)
#   9.  LBR Pinball                 (RSI-3 of ROC-3)
#  10.  Phase-Space Winding         (W — atan2 winding number)
#  11.  Spectral Signature          (PSR — low-freq power ratio via FFT)
#  12.  Modified True Strength      (MTSI — Log(Close/VWAP) double-EMA)
#  13.  Directional Persistence     (Z-score of up/down streak lengths)
#  14.  RQA Determinism             (DET — diagonal-line recurrence ratio)
#
# All indicators are fed through the LENS_10 / LENS_90 z-score pipeline
# (z, z_slope, z_sos) identical to the original trend tester, plus
# WIN_10 / WIN_60 rolling-pct variants where the indicator is unbounded.
#
# Architecture changes vs v4.30:
#   • Blocks 0-1   unchanged  (GPU setup, imports, symbols)
#   • Block  2     extended   (new Numba kernels for momentum math)
#   • Block  3     replaced   (generate_momentum_features replaces trend factory)
#   • Blocks 4-9   unchanged  (loader, audit, sovereign hunt, reporting)
#   • BRAIN_LOCKS  reset      (momentum-specific; re-populated by audit)
# ==============================================================================


# ==============================================================================
# ### BLOCK 0: GPU SETUP
# ==============================================================================
import os, gc, warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU — running CPU. Colab: Runtime → Change runtime type → T4 GPU")
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
DEVICE = '/device:GPU:0' if GPU_AVAILABLE else '/cpu:0'
print(f"[SYSTEM] Active compute device: {DEVICE}\n")


# ==============================================================================
# ### BLOCK 1: SYSTEM INITIALIZATION
# ==============================================================================
import numpy as np, pandas as pd, yfinance as yf
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import random
from numba import jit
from datetime import datetime
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive', force_remount=True)
    OUTPUT_DRIVE_DIR = '/content/drive/MyDrive/judicial_results/Sovereign_Momentum_v1/'
except Exception:
    OUTPUT_DRIVE_DIR = './judicial_results/'

TEST_NAME = "Sovereign_Titan_Momentum_v1.0"
os.makedirs(OUTPUT_DRIVE_DIR, exist_ok=True)

N_FEATURE_WORKERS = max(1, (os.cpu_count() or 2))
print(f"[SYSTEM] Feature generation workers: {N_FEATURE_WORKERS}")

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
    'XLRE','XLU','XLV','XLY','XOM','XOP','XRT'
]


# ==============================================================================
# ### BLOCK 2: NUMBA JIT KERNELS  (trend originals + momentum additions)
# ==============================================================================

# ── Originals kept for compatibility ──────────────────────────────────────────
@jit(nopython=True, cache=True)
def _lin_slope_nb(y):
    n = len(y)
    if n < 2: return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = 0.0
    for i in range(n): y_mean += y[i]
    y_mean /= n
    num = 0.0; den = 0.0
    for i in range(n):
        dx = i - x_mean
        num += dx * (y[i] - y_mean)
        den += dx * dx
    return num / den if den != 0.0 else 0.0

@jit(nopython=True, cache=True)
def _rolling_linslope(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _lin_slope_nb(arr[i - window + 1 : i + 1])
    return out

@jit(nopython=True, cache=True)
def _rolling_wma(arr, window):
    n = len(arr); out = np.full(n, np.nan)
    w_sum = window * (window + 1) / 2.0
    for i in range(window - 1, n):
        s = 0.0
        for j in range(window):
            s += arr[i - window + 1 + j] * (j + 1)
        out[i] = s / w_sum
    return out

@jit(nopython=True, cache=True)
def _kalman_numba(price, r=0.0001, q=0.001):
    x_hat = np.zeros_like(price); p = np.zeros_like(price)
    x_hat[0] = price[0]; p[0] = 1.0
    for t in range(1, len(price)):
        p_minus  = p[t-1] + q
        k        = p_minus / (p_minus + r)
        x_hat[t] = x_hat[t-1] + k * (price[t] - x_hat[t-1])
        p[t]     = (1 - k) * p_minus
    return x_hat

# ── NEW: Wilder RSI ───────────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _rsi_nb(prices, period=14):
    """Wilder's RSI — returns array same length as prices."""
    n = len(prices)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    avg_gain = 0.0; avg_loss = 0.0
    for i in range(1, period + 1):
        d = prices[i] - prices[i - 1]
        if d > 0: avg_gain += d
        else:     avg_loss += -d
    avg_gain /= period; avg_loss /= period
    rs = avg_gain / (avg_loss + 1e-9)
    rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        d = prices[i] - prices[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rs = avg_gain / (avg_loss + 1e-9)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi

# ── NEW: Hurst R/S on ROC series ──────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _hurst_rs_nb(y):
    """Proper rescaled-range Hurst exponent.  H>0.5 = persistent."""
    n = len(y)
    if n < 8: return 0.5
    mean = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    # Cumulative deviations
    cum = 0.0; cmin = 0.0; cmax = 0.0
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
        out[i] = _hurst_rs_nb(arr[i - window + 1 : i + 1])
    return out

# ── NEW: Autocorrelation Half-Life (MHL) ─────────────────────────────────────
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
        rho = cov / var
        if rho <= threshold:
            return float(lag)
    return float(max_lag)

@jit(nopython=True, cache=True)
def _rolling_mhl(arr, window, max_lag=30):
    n = len(arr); out = np.full(n, float(max_lag))
    for i in range(window - 1, n):
        out[i] = _mhl_nb(arr[i - window + 1 : i + 1], max_lag)
    return out

# ── NEW: RQA Determinism ──────────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _rqa_det_nb(y, eps_factor=0.15):
    """DET = diagonal-line points (len≥2) / total recurrent points."""
    n = len(y)
    if n < 5: return 0.5
    mean = 0.0
    for i in range(n): mean += y[i]
    mean /= n
    var = 0.0
    for i in range(n): var += (y[i] - mean) ** 2
    eps = eps_factor * (var / n) ** 0.5
    if eps < 1e-10: return 0.5
    total = 0; diag = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y[i] - y[j]) < eps:
                total += 1
                if i > 0 and j > 0:
                    if abs(y[i - 1] - y[j - 1]) < eps:
                        diag += 1
    if total == 0: return 0.0
    return diag / total

@jit(nopython=True, cache=True)
def _rolling_rqa_det(arr, window):
    n = len(arr); out = np.full(n, 0.5)
    for i in range(window - 1, n):
        out[i] = _rqa_det_nb(arr[i - window + 1 : i + 1])
    return out

# ── NEW: Geometric Curvature κ ────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _rolling_geom_curv(velocity, acceleration):
    """κ = |y''| / (1 + y'²)^1.5  applied element-wise."""
    n = len(velocity); out = np.zeros(n)
    for i in range(n):
        vp = velocity[i]; ap = acceleration[i]
        denom = (1.0 + vp * vp) ** 1.5
        out[i] = abs(ap) / (denom + 1e-12)
    return out

# ── NEW: Quadratic Curvature γ (rolling t² regression) ───────────────────────
@jit(nopython=True, cache=True)
def _quad_gamma_nb(y):
    """OLS coefficient of t² when regressing y on [1, t, t²].
    Solves 3×3 system via Gaussian elimination (Numba-safe)."""
    n = len(y)
    if n < 6: return 0.0
    s1 = 0.0; st = 0.0; st2 = 0.0; st3 = 0.0; st4 = 0.0
    sy = 0.0; sty = 0.0; st2y = 0.0
    for i in range(n):
        t = float(i); t2 = t * t
        s1 += 1.0; st += t; st2 += t2; st3 += t2 * t; st4 += t2 * t2
        sy += y[i]; sty += t * y[i]; st2y += t2 * y[i]
    # Augmented matrix [A | b]
    A = np.zeros((3, 4))
    A[0,0]=s1;  A[0,1]=st;  A[0,2]=st2; A[0,3]=sy
    A[1,0]=st;  A[1,1]=st2; A[1,2]=st3; A[1,3]=sty
    A[2,0]=st2; A[2,1]=st3; A[2,2]=st4; A[2,3]=st2y
    # Forward elimination with partial pivoting
    for col in range(3):
        max_val = abs(A[col, col]); max_row = col
        for row in range(col + 1, 3):
            if abs(A[row, col]) > max_val:
                max_val = abs(A[row, col]); max_row = row
        if max_row != col:
            for k in range(4):
                tmp = A[col, k]; A[col, k] = A[max_row, k]; A[max_row, k] = tmp
        if abs(A[col, col]) < 1e-12: return 0.0
        for row in range(col + 1, 3):
            f = A[row, col] / A[col, col]
            for k in range(col, 4):
                A[row, k] -= f * A[col, k]
    # Back substitution
    x = np.zeros(3)
    for i in range(2, -1, -1):
        x[i] = A[i, 3]
        for j in range(i + 1, 3):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i] if abs(A[i, i]) > 1e-12 else 1.0
    return x[2]   # coefficient of t²

@jit(nopython=True, cache=True)
def _rolling_quad_gamma(arr, window):
    n = len(arr); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _quad_gamma_nb(arr[i - window + 1 : i + 1])
    return out

# ── NEW: Phase-Space Winding ──────────────────────────────────────────────────
@jit(nopython=True, cache=True)
def _winding_nb(x_z, y_z):
    """Winding number W = Σ Δθ / 2π  in phase space (price_z, roc_z)."""
    n = len(x_z)
    if n < 3: return 0.0
    total = 0.0
    pi2 = 2.0 * 3.141592653589793
    for i in range(1, n):
        dtheta = np.arctan2(y_z[i], x_z[i]) - np.arctan2(y_z[i-1], x_z[i-1])
        # Wrap to [-π, π]
        while dtheta >  3.141592653589793: dtheta -= pi2
        while dtheta < -3.141592653589793: dtheta += pi2
        total += dtheta
    return total / pi2

@jit(nopython=True, cache=True)
def _rolling_phase_winding(price_z, roc_z, window):
    n = len(price_z); out = np.full(n, 0.0)
    for i in range(window - 1, n):
        out[i] = _winding_nb(
            price_z[i - window + 1 : i + 1],
            roc_z  [i - window + 1 : i + 1],
        )
    return out

# ── NEW: Directional Persistence streak counter ───────────────────────────────
@jit(nopython=True, cache=True)
def _streak_zscore(returns, window):
    """Count consecutive same-sign bars → z-score vs rolling history."""
    n = len(returns); streaks = np.zeros(n); out = np.zeros(n)
    s = 1
    for i in range(n):
        if i == 0:
            s = 1
        else:
            prev_sign = 1 if returns[i-1] > 0 else (-1 if returns[i-1] < 0 else 0)
            curr_sign = 1 if returns[i]   > 0 else (-1 if returns[i]   < 0 else 0)
            if curr_sign != 0 and curr_sign == prev_sign:
                s += 1
            else:
                s = 1
        streaks[i] = float(s)
    for i in range(window - 1, n):
        w = streaks[i - window + 1 : i + 1]
        mean = 0.0
        for v in w: mean += v
        mean /= window
        var = 0.0
        for v in w: var += (v - mean) ** 2
        std = (var / window) ** 0.5
        out[i] = (streaks[i] - mean) / (std + 1e-9)
    return out

# ── Warm-up: compile all kernels before the run ───────────────────────────────
def _warm_up_numba():
    d  = np.random.randn(100).astype(np.float64)
    d2 = np.random.randn(100).astype(np.float64)
    _rolling_linslope(d, 10)
    _rolling_wma(d, 10)
    _kalman_numba(d)
    _rsi_nb(d, 14)
    _rolling_hurst_rs(d, 40)
    _rolling_mhl(d, 63)
    _rolling_rqa_det(d, 40)
    _rolling_geom_curv(d, d2)
    _rolling_quad_gamma(d, 30)
    _rolling_phase_winding(d, d2, 20)
    _streak_zscore(d, 30)
    print("✅ Numba momentum kernels compiled and ready")

_warm_up_numba()


# ==============================================================================
# ### BLOCK 3: MOMENTUM FEATURE FACTORY
# Replaces generate_factory_features_v2 from the Trend Tester.
# Output columns follow the same LENS_<window>_<name>_z / z_slope / z_sos
# and WIN_<window>_<name>_pct convention so Blocks 4-9 need zero changes.
# ==============================================================================

def _psr_numpy(roc_arr, window=126, f_low_period=63):
    """Power Spectral Ratio: fraction of power at f ≤ 1/63 cycles."""
    n = len(roc_arr); out = np.full(n, np.nan)
    f_low = 1.0 / f_low_period
    for i in range(window - 1, n):
        seg = roc_arr[i - window + 1 : i + 1]
        seg = seg - seg.mean()
        psd = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(window)
        total = psd.sum()
        out[i] = psd[freqs <= f_low].sum() / total if total > 1e-12 else 0.5
    return out


def generate_momentum_features(df):
    """
    Drop-in replacement for generate_factory_features_v2.
    Computes all 14 momentum measures, applies LENS_10/LENS_90 z-pipelines
    and WIN_10/WIN_60 rolling-pct variants.
    """
    df = df.copy()
    df['hlc3']    = (df['high'] + df['low'] + df['close']) / 3
    df['T_FINAL'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    cl  = df['close'].values.astype(np.float64)
    hi  = df['high'].values.astype(np.float64)
    lo  = df['low'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    idx = df.index
    n   = len(cl)

    cl_s  = pd.Series(cl,  index=idx)
    hi_s  = pd.Series(hi,  index=idx)
    lo_s  = pd.Series(lo,  index=idx)
    vol_s = pd.Series(vol, index=idx)

    # ------------------------------------------------------------------
    # 1. CHANDE TREND METER  (CTM)
    # ------------------------------------------------------------------
    # 12 BB %B components (4 periods × High / Low / Close)
    ctm_raw = np.zeros(n)
    for period in [20, 50, 75, 100]:
        for series, sarr in [(cl_s, cl), (hi_s, hi), (lo_s, lo)]:
            sma = series.rolling(period).mean().values
            std = series.rolling(period).std().values
            upper = sma + 2 * std; lower = sma - 2 * std
            bb_pct_b = (sarr - lower) / (upper - lower + 1e-9) * 10
            ctm_raw += bb_pct_b
    # Z-Score component (scaled ×10)
    sma100 = cl_s.rolling(100).mean().values
    std100 = cl_s.rolling(100).std().values
    ctm_raw += (cl - sma100) / (std100 + 1e-9) * 10
    # RSI(14) / 10
    rsi14 = _rsi_nb(cl, 14)
    ctm_raw += rsi14 / 10.0
    # Price Channel ×10
    lo2 = lo_s.rolling(2).min().values
    hi2 = hi_s.rolling(2).max().values
    ctm_raw += (cl - lo2) / (hi2 - lo2 + 1e-9) * 10
    # Rolling min-max scale → 0-100
    ctm_s    = pd.Series(ctm_raw, index=idx)
    roll_min = ctm_s.rolling(252, min_periods=1).min().values
    roll_max = ctm_s.rolling(252, min_periods=1).max().values
    ctm      = (ctm_raw - roll_min) / (roll_max - roll_min + 1e-9) * 100

    # ------------------------------------------------------------------
    # 2. RAW ROC / PRICE MOMENTUM
    # ------------------------------------------------------------------
    roc_2  = cl_s.pct_change(2).fillna(0).values  * 100
    roc_3  = cl_s.pct_change(3).fillna(0).values  * 100
    roc_30 = cl_s.pct_change(30).fillna(0).values * 100
    # Log-return variants
    log_cl  = np.log(cl + 1e-9)
    roc_2_log  = np.diff(log_cl, n=2, prepend=[log_cl[0], log_cl[0]])
    roc_3_log  = np.diff(log_cl, n=3, prepend=[log_cl[0]]*3)
    roc_30_log = np.diff(log_cl, n=30, prepend=[log_cl[0]]*30)

    # ------------------------------------------------------------------
    # 3. AUTOCORRELATION HALF-LIFE  (MHL on 63-bar rolling ROC-3 window)
    # ------------------------------------------------------------------
    mhl = _rolling_mhl(roc_3, window=63, max_lag=30)

    # ------------------------------------------------------------------
    # 4. HURST EXPONENT ON ROC  (R/S, 63-bar window)
    # ------------------------------------------------------------------
    hurst_roc = _rolling_hurst_rs(roc_3, window=63)

    # ------------------------------------------------------------------
    # 5. GARR RATIO  (geometric mean 1-mo / 12-mo)
    # ------------------------------------------------------------------
    log_ret    = np.log(cl_s / cl_s.shift(1)).fillna(0)
    # GARR_21 = exp(mean log-ret over 21 bars) - 1
    garr_21  = log_ret.rolling(21).mean().apply(np.exp) - 1
    # GARR_252 = exp(mean log-ret over 252 bars) - 1
    garr_252 = log_ret.rolling(252).mean().apply(np.exp) - 1
    garr_ratio = (garr_21 / (garr_252.abs() + 1e-9) *
                  garr_252.apply(np.sign)).values

    # ------------------------------------------------------------------
    # 6. ROC ACCELERATION & JERK  (fast=7, medium=14, slow=21)
    # ------------------------------------------------------------------
    roc_7  = cl_s.pct_change(7).fillna(0).values  * 100
    roc_14 = cl_s.pct_change(14).fillna(0).values * 100
    roc_21 = cl_s.pct_change(21).fillna(0).values * 100
    accel_7_21  = roc_7  - roc_21   # fast vs slow
    accel_7_14  = roc_7  - roc_14   # fast vs medium
    accel_14_21 = roc_14 - roc_21   # medium vs slow
    jerk        = np.diff(accel_7_21,  prepend=accel_7_21[0])
    jerk_med    = np.diff(accel_7_14,  prepend=accel_7_14[0])

    # ------------------------------------------------------------------
    # 7. GEOMETRIC CURVATURE  κ
    # ------------------------------------------------------------------
    velocity     = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)   # 1-period ROC
    acceleration = np.diff(velocity, prepend=velocity[0])
    geom_curv    = _rolling_geom_curv(velocity, acceleration)

    # ------------------------------------------------------------------
    # 8. QUADRATIC CURVATURE  γ  (250-bar rolling t² regression)
    # ------------------------------------------------------------------
    daily_ret    = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)
    quad_gamma   = _rolling_quad_gamma(daily_ret, window=250)

    # ------------------------------------------------------------------
    # 9. LBR PINBALL  (RSI-3 of ROC-3)
    # ------------------------------------------------------------------
    lbr_pinball = _rsi_nb(roc_3, period=3)

    # ------------------------------------------------------------------
    # 10. PHASE-SPACE WINDING  W  (price z × ROC z, 20-bar window)
    # ------------------------------------------------------------------
    def _zscore_rolling(arr, w=20):
        s = pd.Series(arr, index=idx)
        rm = s.rolling(w).mean().values
        rs = s.rolling(w).std().values
        return (arr - rm) / (rs + 1e-9)

    price_z20 = _zscore_rolling(cl,    20).astype(np.float64)
    roc_z20   = _zscore_rolling(roc_3, 20).astype(np.float64)
    phase_wind = _rolling_phase_winding(price_z20, roc_z20, window=20)

    # ------------------------------------------------------------------
    # 11. SPECTRAL SIGNATURE  PSR  (FFT, 126-bar window, f_low=1/63)
    # ------------------------------------------------------------------
    psr = _psr_numpy(roc_3, window=126, f_low_period=63)
    psr = np.nan_to_num(psr, nan=0.5)

    # ------------------------------------------------------------------
    # 12. MODIFIED TRUE STRENGTH INDEX  MTSI-v2
    # ------------------------------------------------------------------
    tp_v     = pd.Series(cl * vol, index=idx)
    vwap_2   = (tp_v.rolling(2).sum() / (vol_s.rolling(2).sum() + 1e-9)).values
    diff_mtsi    = np.log(cl / (np.abs(vwap_2) + 1e-9))
    diff_s       = pd.Series(diff_mtsi, index=idx)
    abs_diff_s   = pd.Series(np.abs(diff_mtsi), index=idx)
    dbl_diff     = diff_s.ewm(span=3, adjust=False).mean().ewm(span=2, adjust=False).mean()
    dbl_abs_diff = abs_diff_s.ewm(span=3, adjust=False).mean().ewm(span=2, adjust=False).mean()
    mtsi         = (100.0 * dbl_diff / (dbl_abs_diff + 1e-9)).values

    # ------------------------------------------------------------------
    # 13. DIRECTIONAL PERSISTENCE  (z-score of up/down streak lengths)
    # ------------------------------------------------------------------
    dir_persist = _streak_zscore(daily_ret, window=63)

    # ------------------------------------------------------------------
    # 14. RQA DETERMINISM  DET  (50-bar window, ε = 0.15 × σ_ROC)
    # ------------------------------------------------------------------
    rqa_det = _rolling_rqa_det(roc_3, window=50)

    # ------------------------------------------------------------------
    # Assemble Z-LENS indicator dictionary
    # Identical structure to the Trend Tester: LENS_10 / LENS_90 applied.
    # Unbounded indicators also get WIN_10 / WIN_60 rolling-pct variants.
    # ------------------------------------------------------------------
    Z_LENS_INDICATORS = {
        # CTM & RSI
        'ctm':          pd.Series(ctm,          index=idx),
        'rsi_14':       pd.Series(rsi14,         index=idx),
        # ROC variants
        'roc_2':        pd.Series(roc_2,         index=idx),
        'roc_3':        pd.Series(roc_3,         index=idx),
        'roc_30':       pd.Series(roc_30,        index=idx),
        'roc_2_log':    pd.Series(roc_2_log,     index=idx),
        'roc_3_log':    pd.Series(roc_3_log,     index=idx),
        'roc_30_log':   pd.Series(roc_30_log,    index=idx),
        # Momentum structure
        'mhl':          pd.Series(mhl,           index=idx),
        'hurst_roc':    pd.Series(hurst_roc,     index=idx),
        'garr_ratio':   pd.Series(garr_ratio,    index=idx),
        # Acceleration family
        'accel_7_21':   pd.Series(accel_7_21,    index=idx),
        'accel_7_14':   pd.Series(accel_7_14,    index=idx),
        'accel_14_21':  pd.Series(accel_14_21,   index=idx),
        'jerk':         pd.Series(jerk,          index=idx),
        'jerk_med':     pd.Series(jerk_med,      index=idx),
        # Curvature
        'geom_curv':    pd.Series(geom_curv,     index=idx),
        'quad_gamma':   pd.Series(quad_gamma,    index=idx),
        # Oscillator / exhaustion
        'lbr_pinball':  pd.Series(lbr_pinball,   index=idx),
        'phase_wind':   pd.Series(phase_wind,    index=idx),
        'psr':          pd.Series(psr,           index=idx),
        'mtsi':         pd.Series(mtsi,          index=idx),
        'dir_persist':  pd.Series(dir_persist,   index=idx),
        'rqa_det':      pd.Series(rqa_det,       index=idx),
    }

    # Apply LENS_10 and LENS_90  (z, z_slope, z_sos)
    for name, ind in Z_LENS_INDICATORS.items():
        arr = ind.values.astype(np.float64)
        for lens in [10, 90]:
            rm   = pd.Series(arr, index=idx).rolling(lens).mean().values
            rs   = pd.Series(arr, index=idx).rolling(lens).std().values
            z    = (arr - rm) / (rs + 1e-9)
            zs   = _rolling_linslope(z, lens)
            zsos = _rolling_linslope(zs, lens)
            df[f'LENS_{lens}_{name}_z']       = z
            df[f'LENS_{lens}_{name}_z_slope'] = zs
            df[f'LENS_{lens}_{name}_z_sos']   = zsos

    # WIN_10 / WIN_60 rolling-pct for unbounded indicators
    WIN_INDICATORS = {
        'mhl':       mhl,
        'hurst_roc': hurst_roc,
        'garr_ratio': garr_ratio,
        'accel_7_21': accel_7_21,
        'jerk':      jerk,
        'geom_curv': geom_curv,
        'quad_gamma': quad_gamma,
        'phase_wind': phase_wind,
        'rqa_det':   rqa_det,
    }
    for name, arr in WIN_INDICATORS.items():
        for win in [10, 60]:
            rm = pd.Series(arr, index=idx).rolling(win).mean().values
            df[f'WIN_{win}_{name}_pct'] = arr / (np.abs(rm) + 1e-9) - 1.0

    return (df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .dropna(subset=['T_FINAL'])
              .fillna(0))


# ==============================================================================
# ### BLOCK 4: PARALLEL LOADER
# ==============================================================================
def fetch_data(symbol):
    try:
        data = yf.download(symbol, period="3y", interval="1d", progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
        return data if 'close' in data.columns else None
    except Exception:
        return None

def load_hybrid_data_parallel(brain_name, symbol_list, dl_workers=20):
    # Guard: generate_momentum_features must exist before we can process symbols.
    if 'generate_momentum_features' not in globals():
        raise RuntimeError(
            "❌ generate_momentum_features is not defined — Block 3 did not "
            "finish executing. Re-run the Block 3 cell before Block 4."
        )

    # Define the worker as a LOCAL closure so it is always available to
    # ThreadPoolExecutor regardless of Colab cell execution order.
    # (ThreadPoolExecutor does not pickle callables, so closures work fine.)
    def _worker(args):
        symbol, raw_dict = args
        try:
            raw_df = pd.DataFrame(raw_dict)
            raw_df.index = pd.to_datetime(raw_df.index)
            if len(raw_df) < 260:
                return None
            processed = generate_momentum_features(raw_df)
            if processed.empty:
                return None
            processed['symbol'] = symbol
            return processed.reset_index()
        except Exception:
            return None

    print(f"📥 Parallel download: {len(symbol_list)} symbols...")
    raw_results = {}
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        fut_map = {pool.submit(fetch_data, sym): sym for sym in symbol_list}
        for fut in tqdm(as_completed(fut_map), total=len(symbol_list),
                        desc="⬇ Downloading"):
            sym  = fut_map[fut]
            data = fut.result()
            if data is not None and len(data) >= 260:
                raw_results[sym] = data
    print(f"   ✅ {len(raw_results)}/{len(symbol_list)} symbols fetched")
    if not raw_results:
        return pd.DataFrame()
    work_items = [(sym, df.to_dict()) for sym, df in raw_results.items()]
    all_data   = []
    # ThreadPoolExecutor: Numba JIT kernels release the GIL, so threads give
    # true CPU parallelism here — same throughput as processes, no pickling.
    print(f"⚙ Building momentum features (threads={N_FEATURE_WORKERS})...")
    with ThreadPoolExecutor(max_workers=N_FEATURE_WORKERS) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(work_items),
                        desc="⚙ Features"):
            try:
                result = fut.result()
                if result is not None:
                    result = result.set_index(result.columns[0])
                    all_data.append(result)
            except Exception as e:
                print(f"  ⚠️  Symbol failed: {e}")
    if not all_data:
        print("❌ No valid data after feature generation.")
        return pd.DataFrame()
    print(f"   ✅ {len(all_data)} symbols processed")
    return pd.concat(all_data, axis=0)


# ==============================================================================
# ### BLOCK 5: GPU-ACCELERATED AUDIT  (unchanged from Trend Tester)
# ==============================================================================
def build_full_model(model_type, n_features, seq_len, device=DEVICE):
    with tf.device(device):
        model = Sequential([
            Input(shape=(seq_len, n_features)),
            GRU(128,  return_sequences=True) if model_type == 'GRU'
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
def _eval_accuracy(model, X_batch, y_batch):
    preds   = tf.squeeze(model(X_batch, training=False), axis=-1)
    correct = tf.equal(tf.cast(preds >= 0.5, tf.int32),
                       tf.cast(y_batch, tf.int32))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def run_judicial_audit(brain_name, master_df, model_type='GRU',
                       seq_len=10, epochs=50, batch_size=2048):
    feature_cols = [c for c in master_df.columns
                    if c.startswith('LENS_') or c.startswith('WIN_')]
    n_features   = len(feature_cols)
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(
        master_df[feature_cols].values
    ).astype(np.float32)
    y_raw    = master_df['T_FINAL'].values.astype(np.float32)
    n        = len(X_scaled)
    X_seqs   = np.stack([X_scaled[i - seq_len:i] for i in range(seq_len, n)])
    y_seqs   = y_raw[seq_len:]
    split        = int(len(X_seqs) * 0.8)
    X_tr, X_val  = X_seqs[:split], X_seqs[split:]
    y_tr, y_val  = y_seqs[:split], y_seqs[split:]
    print(f"  [DATA] train={len(X_tr):,}  val={len(X_val):,}  features={n_features}")
    AUTO     = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(min(20_000, len(X_tr)), reshuffle_each_iteration=True)
                .batch(batch_size).prefetch(AUTO))
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(batch_size * 2).prefetch(AUTO))
    model = build_full_model(model_type, n_features, seq_len)
    with tf.device(DEVICE):
        model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=1e-5),
            ],
            verbose=1,
        )
    X_val_tf     = tf.constant(X_val)
    y_val_tf     = tf.constant(y_val)
    baseline_acc = _eval_accuracy(model, X_val_tf, y_val_tf).numpy()
    print(f"  [MODEL] Baseline val accuracy: {baseline_acc:.4f}")
    report_rows = []
    for fi, feat_name in enumerate(tqdm(feature_cols, desc="Permutation scoring")):
        try:
            X_perm = X_val.copy()
            flat   = X_perm[:, :, fi].flatten()
            np.random.shuffle(flat)
            X_perm[:, :, fi] = flat.reshape(X_perm[:, :, fi].shape)
            perm_acc = _eval_accuracy(model, tf.constant(X_perm), y_val_tf).numpy()
            report_rows.append({
                'Feature': feat_name,
                'I_raw':   max(0.0, baseline_acc - perm_acc),
            })
        except Exception:
            report_rows.append({'Feature': feat_name, 'I_raw': 0.0})
    del model; gc.collect(); tf.keras.backend.clear_session()
    return pd.DataFrame(report_rows)


# ==============================================================================
# ### BLOCK 6: SOVEREIGN HUNT & DIVERSITY ANCHORS
# BRAIN_LOCKS start empty — they will be auto-populated after the first full run.
# Momentum brain split:
#   DIRECTION → persistent directional momentum (CTM, Hurst-ROC, GARR, MHL)
#   EASE      → smooth clean signals (PSR, MTSI, Directional Persistence, LBR)
#   EXP       → explosive / exhaustion (Accel, Jerk, Geometric κ, Phase-Winding)
# ==============================================================================
BRAIN_LOCKS = {
    'DIRECTION': [],
    'EASE':      [],
    'EXP':       [],
}

def _parse_feature_name(f):
    TRANSFORM_TOKENS = {'z', 'slope', 'sos', 'pct'}
    if f.startswith('LENS_') or f.startswith('WIN_'):
        parts     = f.split('_')
        prefix    = parts[0]
        window    = parts[1]
        remainder = list(parts[2:])
        while remainder and remainder[-1] in TRANSFORM_TOKENS:
            remainder.pop()
        indicator = '_'.join(remainder)
        family    = '_'.join(p for p in remainder if not p.isdigit())
        lookback  = f'{prefix}_{window}_{indicator}'
        return prefix, window, lookback, family
    return None, None, f, f

def apply_sovereign_hunt(ledger_df, master_data_df, brain_name, max_slots=19):
    locked_list = BRAIN_LOCKS.get(brain_name, [])
    candidates  = ledger_df.sort_values(by='I_raw', ascending=False)
    picked      = [f for f in locked_list if f in ledger_df['Feature'].values]
    for lf in locked_list:
        if lf not in ledger_df['Feature'].values:
            print(f"  ⚠️  BRAIN_LOCK '{lf}' not found — check name")
    CORR_THRESHOLD = 0.85
    family_lookback = {}
    for f in picked:
        _, _, lookback, family = _parse_feature_name(f)
        if family not in family_lookback:
            family_lookback[family] = lookback
    feat_cols   = [c for c in master_data_df.columns
                   if c.startswith('LENS_') or c.startswith('WIN_')]
    corr_matrix = master_data_df[feat_cols].corr()
    for _, row in candidates.iterrows():
        if len(picked) >= max_slots:
            break
        f_name = row['Feature']
        if f_name in picked:
            continue
        _, _, f_lookback, f_family = _parse_feature_name(f_name)
        if f_family in family_lookback and family_lookback[f_family] != f_lookback:
            continue
        if (len(picked) > 0 and
                corr_matrix[f_name].loc[picked].max() > CORR_THRESHOLD):
            continue
        picked.append(f_name)
        if f_family not in family_lookback:
            family_lookback[f_family] = f_lookback
    pca = PCA()
    pca.fit(RobustScaler().fit_transform(master_data_df[picked]))
    return picked, np.cumsum(pca.explained_variance_ratio_)

def generate_judicial_ledger(brain_name, report_df, master_data_df, iteration=1):
    df           = report_df.copy()
    df['I_Norm'] = (df['I_raw'] - df['I_raw'].min()) / \
                   (df['I_raw'].max() - df['I_raw'].min() + 1e-9)
    active_picks, var_map = apply_sovereign_hunt(df, master_data_df, brain_name)
    corr_sub = master_data_df[active_picks].corr().abs()
    avg_corr = ((corr_sub.sum().sum() - len(active_picks)) /
                (len(active_picks)**2 - len(active_picks) + 1e-9))
    print(f"\n╔══ {brain_name} SOVEREIGN MOMENTUM CORE v1.0 (Iter {iteration}) ══╗")
    print(f"║ {'RNK':<3} | {'MOMENTUM FEATURE':<35} | {'UV%':<4} | {'mR':<4} | {'IMPACT':<8} ║")
    print("╠" + "═"*4 + "╬" + "═"*37 + "╬" + "═"*6 + "╬" + "═"*6 + "╬" + "═"*10 + "╣")
    for i, f_name in enumerate(active_picks):
        f_row       = df[df['Feature'] == f_name].iloc[0]
        is_locked   = f_name in BRAIN_LOCKS.get(brain_name, [])
        icon        = "🔒" if is_locked else "🔭"
        other_picks = [p for p in active_picks if p != f_name]
        max_r  = corr_sub[f_name].loc[other_picks].max() if other_picks else 0.0
        uv_val = ((1 - corr_sub[f_name].loc[other_picks].mean()) * 100
                  if other_picks else 100.0)
        print(f"║ {i+1:02d}  | {icon} {f_name[:33]:<33} | "
              f"{uv_val:>3.0f}% | {max_r:.2f} | {f_row['I_Norm']:.4f} ║")
        df.loc[df['Feature'] == f_name,
               ['UV%', 'Max_R', 'Is_Locked']] = [uv_val, max_r, is_locked]
    total_var = var_map[-1] if len(var_map) > 0 else 0
    print("╠" + "═"*73 + "╣")
    print(f"║ PCA TOTAL VARIANCE RETENTION: {total_var*100:>33.2f}% ║")
    print(f"║ AVG TEAM CROSS-CORRELATION:   {avg_corr:>35.3f} ║")
    print(f"║ SLOTS FILLED:                 {len(active_picks):>35}/19 ║")
    print("╚" + "═"*73 + "╝")
    return df[df['Feature'].isin(active_picks)]


# ==============================================================================
# ### BLOCK 7: COMMAND CENTER
# ==============================================================================
print("\n--- SOVEREIGN TITAN v1.0 — MOMENTUM EDITION ---")
choice        = input("Select Brain (1:DIRECTION / 2:EASE / 3:EXP / 4:ALL): ")
BRAINS_TO_RUN = (['DIRECTION', 'EASE', 'EXP'] if choice == '4'
                 else [{'1': 'DIRECTION', '2': 'EASE', '3': 'EXP'}[choice]])
num_symbols   = int(input("Symbols per iteration (Default 75): ") or "75")
num_iters     = int(input("Iterations to run (Default 20): ")     or "20")

final_report_accumulator = []

for BRAIN in BRAINS_TO_RUN:
    CURRENT_MODEL_TYPE = 'GRU' if BRAIN == 'DIRECTION' else 'LSTM'
    print(f"\n[SYSTEM] Brain: {BRAIN} | Model: {CURRENT_MODEL_TYPE} | Device: {DEVICE}")

    for it in range(1, num_iters + 1):
        print(f"\n{'─'*55}")
        print(f"  Iteration {it}/{num_iters}  —  Brain: {BRAIN}")
        print(f"{'─'*55}")
        POOL      = random.sample(TITAN_SYMBOLS, min(num_symbols, len(TITAN_SYMBOLS)))
        master_df = load_hybrid_data_parallel(BRAIN, POOL)
        if master_df.empty:
            print("  ⚠️  Empty master_df — skipping.")
            continue
        report_raw       = run_judicial_audit(BRAIN, master_df,
                                              model_type=CURRENT_MODEL_TYPE)
        iteration_ledger = generate_judicial_ledger(BRAIN, report_raw,
                                                    master_df, iteration=it)
        iteration_ledger['Iteration']  = it
        iteration_ledger['Brain']      = BRAIN
        iteration_ledger['Model_Type'] = CURRENT_MODEL_TYPE
        iteration_ledger['Timestamp']  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_report_accumulator.append(iteration_ledger)
        gc.collect()
        tf.keras.backend.clear_session()


# ==============================================================================
# ### BLOCK 8: FINAL EXPORT & SOVEREIGN SELECTION  (unchanged)
# ==============================================================================
if final_report_accumulator:
    raw_df = pd.concat(final_report_accumulator, axis=0)
    stats  = (raw_df.groupby(['Brain', 'Feature'])
              .agg(Persistence=('Feature', 'count'),
                   A_Impact=('I_Norm', 'mean'),
                   A_UV=('UV%', 'mean'))
              .reset_index())
    final_df = (raw_df.merge(stats, on=['Brain', 'Feature'], how='left')
                      .sort_values(['Brain', 'Persistence', 'A_Impact'],
                                   ascending=False))
    report_filename = (f"Momentum_Audit_Master_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    report_path = os.path.join(OUTPUT_DRIVE_DIR, report_filename)
    final_df.to_csv(report_path, index=False)
    print("\n" + "="*65)
    print("✅ GLOBAL MOMENTUM AUDIT COMPLETE")
    print(f"📊 DATA ROWS COLLECTED: {len(raw_df)}")
    print(f"📂 CSV SAVED TO:        {report_path}")
    print("="*65)
    print("\n" + "═"*65)
    print("🚀 FINAL SOVEREIGN ARRAYS (TOP 19 PER BRAIN)")
    print("═"*65)
    FINAL_SELECTIONS = {}
    for brain in BRAINS_TO_RUN:
        brain_stats = (stats[stats['Brain'] == brain]
                       .sort_values(['Persistence', 'A_Impact'], ascending=False))
        top_19 = brain_stats.head(19)
        FINAL_SELECTIONS[brain] = top_19['Feature'].tolist()
        print(f"\n💎 FINAL 19 — BRAIN: {brain}")
        print(f"{'RNK':<3} | {'FEATURE':<38} | {'PERSIST':<8} | {'AVG_IMP':<8}")
        print("─" * 62)
        for i, row in top_19.reset_index(drop=True).iterrows():
            print(f"{i+1:02d}  | {row['Feature']:<38} | "
                  f"{int(row['Persistence']):>2}/{num_iters:<5} | "
                  f"{row['A_Impact']:.4f}")
    for brain, winners in FINAL_SELECTIONS.items():
        BRAIN_LOCKS[brain] = winners
    print("\n" + "═"*65)
    print("✅ FINAL 19 SYNCED TO BRAIN_LOCKS")
    print(f"📂 TOTAL UNIQUE FEATURES LOGGED: {len(stats)}")
    print("═"*65)
else:
    print("\n⚠️ [CRITICAL] No data collected. Audit failed.")


# ==============================================================================
# ### BLOCK 9: FINAL AUDIT — deep convergence analysis  (unchanged)
# ==============================================================================
if final_report_accumulator:
    print("\n" + "═"*70)
    print("  BLOCK 9 — MOMENTUM AUDIT RUNNING ON GENERATED CSV")
    print(f"  Source: {report_path}")
    print("═"*70)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from scipy.stats import spearmanr, kendalltau
    try:
        from IPython.display import display, HTML
    except ImportError:
        pass
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                         "axes.spines.right": False})
    _N_FEATURES = 19
    _df = final_df.copy()
    _df["Iteration"] = pd.to_numeric(_df["Iteration"],  errors="coerce").fillna(1).astype(int)
    _df["Is_Locked"] = _df["Is_Locked"].astype(str).str.lower().isin(["true","1","yes"])
    for col in ["I_Norm", "A_Impact", "A_UV", "UV%", "Max_R"]:
        if col in _df.columns:
            _df[col] = pd.to_numeric(_df[col], errors="coerce")
    _BRAINS  = sorted(_df["Brain"].unique())
    _ITERS   = sorted(_df["Iteration"].unique())
    _N_ITERS = len(_ITERS)
    print(f"  Rows       : {len(_df):,}")
    print(f"  Brains     : {_BRAINS}")
    print(f"  Iterations : {_ITERS}  (n={_N_ITERS})")
    print(f"  Unique feat: {_df['Feature'].nunique():,}")

    def _audit_rank_iter(brain_df, iteration):
        sub = brain_df[brain_df["Iteration"] == iteration].copy()
        score_col = "A_Impact" if sub["A_Impact"].notna().any() else "I_Norm"
        sub = sub.groupby("Feature")[score_col].mean().reset_index()
        sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)
        return pd.Series(sub.index.values + 1, index=sub["Feature"].values)

    def _audit_iter_rank_corr(brain_df):
        rows      = []
        all_feats = sorted(brain_df["Feature"].unique())
        for i in range(len(_ITERS) - 1):
            ia, ib = _ITERS[i], _ITERS[i + 1]
            ra = _audit_rank_iter(brain_df, ia).reindex(all_feats).fillna(len(all_feats) + 1)
            rb = _audit_rank_iter(brain_df, ib).reindex(all_feats).fillna(len(all_feats) + 1)
            sp, _ = spearmanr(ra, rb)
            kt, _ = kendalltau(ra, rb)
            top_a = set(ra.nsmallest(_N_FEATURES).index)
            top_b = set(rb.nsmallest(_N_FEATURES).index)
            rows.append(dict(
                iter_a=ia, iter_b=ib,
                spearman_r=round(sp, 4), kendall_tau=round(kt, 4),
                top19_overlap=len(top_a & top_b),
                top19_new_entries=len(top_b - top_a),
            ))
        return pd.DataFrame(rows)

    def _audit_sov_score_cv(brain_df):
        score_col = "A_Impact" if brain_df["A_Impact"].notna().any() else "I_Norm"
        grp = (brain_df.groupby("Feature")[score_col]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "avg", "std": "sd", "count": "n_obs"}))
        grp["cv"] = (grp["sd"] / grp["avg"].replace(0, np.nan)).fillna(0)
        return grp.sort_values("avg", ascending=False)

    def _audit_verdict(corr_df, cv_df, brain):
        base = dict(brain=brain, n_iters=_N_ITERS, last_rho=0.0,
                    last_overlap=0, median_cv=0.0, trending_up=False,
                    pass_rho=False, pass_overlap=False, pass_cv=False,
                    notes=[], recommend_n=max(_N_ITERS + 3, 8))
        if corr_df.empty:
            base["verdict"] = "CANNOT_ASSESS"
            base["reason"]  = "Only 1 iteration found."
            return base
        last      = corr_df.iloc[-1]
        rho       = last["spearman_r"]
        overlap   = last["top19_overlap"]
        median_cv = cv_df["cv"].median()
        pass_rho  = rho >= 0.85
        pass_ovl  = overlap >= 16
        pass_cv   = median_cv <= 0.20
        all_pass  = pass_rho and pass_ovl and pass_cv
        notes = []
        if not pass_rho: notes.append(f"rank ρ={rho:.3f} < 0.85 — rankings still shifting")
        if not pass_ovl: notes.append(f"top-19 overlap={overlap}/19 < 16 — slot instability")
        if not pass_cv:  notes.append(f"median CV={median_cv:.3f} > 0.20 — score variance too high")
        trending_up = (len(corr_df) >= 2 and
                       corr_df["spearman_r"].iloc[-1] > corr_df["spearman_r"].iloc[-2])
        return dict(
            verdict      = "✅ SUFFICIENT" if all_pass else "⚠️  MORE NEEDED",
            brain        = brain, n_iters=_N_ITERS,
            last_rho     = rho, last_overlap=overlap,
            median_cv    = median_cv, trending_up=trending_up,
            pass_rho     = pass_rho, pass_overlap=pass_ovl, pass_cv=pass_cv,
            notes        = notes,
            recommend_n  = max(_N_ITERS + 3, 8) if not all_pass else _N_ITERS,
        )

    def _audit_build_final_roster(brain_df, n=_N_FEATURES):
        score_col = "A_Impact" if brain_df["A_Impact"].notna().any() else "I_Norm"
        grp = (brain_df.groupby("Feature")
               .agg(Runs=("Iteration","nunique"), Score=(score_col,"mean"),
                    Avg_INorm=("I_Norm","mean"), Avg_UV=("UV%","mean"),
                    Is_Locked=("Is_Locked","first"))
               .reset_index())
        mask = (grp["Feature"].str.contains("LENS_", case=False, na=False) |
                grp["Feature"].str.contains("WIN_",  case=False, na=False))
        grp = grp[mask].copy()
        grp["Persistence_pct"]  = grp["Runs"] / _N_ITERS
        grp["Stability_Weight"] = grp["Persistence_pct"] ** 2
        grp["Weighted_Score"]   = grp["Score"] * grp["Stability_Weight"]
        locked   = grp[grp["Is_Locked"]].copy()
        unlocked = grp[~grp["Is_Locked"]].sort_values("Weighted_Score", ascending=False)
        final = pd.concat([locked, unlocked.head(n - len(locked))], ignore_index=True)
        final = final.sort_values("Weighted_Score", ascending=False).reset_index(drop=True)
        final.index = final.index + 1
        return final

    _corr_tables   = {}; _cv_tables = {}; _verdicts = {}; _final_rosters = {}
    for brain in _BRAINS:
        bdf = _df[_df["Brain"] == brain]
        ct  = _audit_iter_rank_corr(bdf); cvt = _audit_sov_score_cv(bdf)
        _corr_tables[brain]   = ct; _cv_tables[brain]     = cvt
        _verdicts[brain]      = _audit_verdict(ct, cvt, brain)
        _final_rosters[brain] = _audit_build_final_roster(bdf)
    print(f"\n✅ Audit engines complete for: {_BRAINS}")

    for brain in _BRAINS:
        corr = _corr_tables[brain]; verd = _verdicts[brain]
        print(f"\n  ┌─ {brain} {'─'*(54-len(brain))}┐")
        if "CANNOT_ASSESS" in verd["verdict"]:
            print(f"  │  Only 1 iteration — rankings established, no trend yet.    │")
            print(f"  └{'─'*57}┘")
        else:
            print(f"  │  {'Iters':>8}  {'ρ Spearman':>11}  {'τ Kendall':>10}  "
                  f"{'Overlap/19':>10}  {'New slots':>9}  │")
            print(f"  │  {'─'*8}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*9}  │")
            for _, r in corr.iterrows():
                flag = " ✓" if r["spearman_r"] >= 0.85 else " ⚠"
                print(f"  │  {int(r.iter_a):>3}→{int(r.iter_b):<4}  "
                      f"{r.spearman_r:>10.4f}  {r.kendall_tau:>10.4f}  "
                      f"{int(r.top19_overlap):>8}/19  {int(r.top19_new_entries):>9}  │{flag}")
            print(f"  └{'─'*57}┘")
        print(f"\n  VERDICT [{brain}]: {verd['verdict']}")
        if "CANNOT_ASSESS" not in verd["verdict"]:
            print(f"    last ρ={verd['last_rho']:.4f}  overlap={verd['last_overlap']}/19"
                  f"  median CV={verd['median_cv']:.4f}"
                  f"  trending={'↑' if verd['trending_up'] else '─'}")
        for note in verd.get("notes", []): print(f"    ⚠  {note}")

    for brain in _BRAINS:
        roster = _final_rosters[brain]
        print(f"\n{'═'*78}")
        print(f"  STABILITY-WEIGHTED ROSTER ─ {brain}")
        print(f"  (Persistence² × raw score — dampens churn, rewards consistency)")
        print(f"{'═'*78}")
        print(f"  {'RNK':<4} {'FEATURE':<40} {'W_SCORE':>7} {'RAW':>7} {'PERS':>6} {'LK':>3}")
        print(f"  {'─'*72}")
        for rank, row in roster.iterrows():
            lock_s = "🔒" if row["Is_Locked"] else "  "
            pers_s = f"{row['Persistence_pct']*100:.0f}%"
            print(f"  {rank:02d}.  {row['Feature']:<40} {row['Weighted_Score']:>7.4f} "
                  f"{row['Score']:>7.4f} {pers_s:>5} {lock_s}")

    n_b = len(_BRAINS)
    if _N_ITERS > 1:
        fig, axes = plt.subplots(1, n_b, figsize=(6 * n_b, 4), squeeze=False)
        for ax, brain in zip(axes[0], _BRAINS):
            corr = _corr_tables[brain]
            if corr.empty:
                ax.text(0.5, 0.5, "Only 1 iteration", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
            else:
                x = [f"{int(r.iter_a)}→{int(r.iter_b)}" for _, r in corr.iterrows()]
                ax.plot(x, corr["spearman_r"], "o-", color="#457b9d",
                        label="Spearman ρ", lw=2)
                ax.plot(x, corr["top19_overlap"] / _N_FEATURES, "s--",
                        color="#e63946", label=f"Top-{_N_FEATURES} overlap", lw=1.5)
                ax.axhline(0.85, ls=":", color="gray", lw=1, label="ρ=0.85 threshold")
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{brain} — Momentum Convergence", fontweight="bold")
                ax.set_ylabel("Score / Overlap fraction")
                ax.set_xlabel("Iteration transition")
                ax.legend(fontsize=8)
                ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Sovereign Momentum — Feature Convergence Across Iterations",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout(); plt.show()

    fig2, axes2 = plt.subplots(1, n_b, figsize=(9 * n_b, 7), squeeze=False)
    for ax, brain in zip(axes2[0], _BRAINS):
        roster = _final_rosters[brain].reset_index(drop=False)
        roster = roster.sort_values("Score", ascending=True)
        colors = ["#e63946" if lk else "#457b9d" for lk in roster["Is_Locked"]]
        ax.barh(roster["Feature"], roster["Score"], color=colors)
        ax.set_xlabel("Score")
        ax.set_title(f"{brain} — Final {_N_FEATURES} Momentum Features", fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(handles=[Patch(color="#e63946", label="Locked"),
                            Patch(color="#457b9d", label="Ranked")],
                  loc="lower right", fontsize=8)
    fig2.suptitle("Final Momentum Feature Rosters", fontsize=13, fontweight="bold", y=1.02)
    fig2.tight_layout(); plt.show()

    if _N_ITERS > 1:
        all_top_feats = set()
        for brain in _BRAINS:
            all_top_feats |= set(_final_rosters[brain]["Feature"])
        cv_matrix = pd.DataFrame(index=sorted(all_top_feats), columns=_BRAINS, dtype=float)
        for brain in _BRAINS:
            cv = _cv_tables[brain].reindex(sorted(all_top_feats))
            cv_matrix[brain] = cv["cv"]
        fig3, ax3 = plt.subplots(
            figsize=(max(5, 3 * n_b), max(8, len(all_top_feats) * 0.38)))
        im = ax3.imshow(cv_matrix.values.astype(float), aspect="auto",
                        cmap="RdYlGn_r", vmin=0, vmax=0.5)
        ax3.set_xticks(range(n_b)); ax3.set_xticklabels(_BRAINS, fontsize=9)
        ax3.set_yticks(range(len(cv_matrix)))
        ax3.set_yticklabels(cv_matrix.index, fontsize=6)
        ax3.set_title("Score CV per Feature per Brain\n(green=stable, red=volatile)",
                      fontweight="bold")
        plt.colorbar(im, ax=ax3, label="CV (σ/μ)"); fig3.tight_layout(); plt.show()

    print("\n" + "═"*70)
    print("  ITERATION SUFFICIENCY SUMMARY")
    print("═"*70)
    all_sufficient = True
    for brain in _BRAINS:
        v = _verdicts[brain]
        all_sufficient = all_sufficient and ("SUFFICIENT" in v["verdict"])
        icon_rho = "✅" if v["pass_rho"]     else "❌"
        icon_ovl = "✅" if v["pass_overlap"] else "❌"
        icon_cv  = "✅" if v["pass_cv"]      else "❌"
        print(f"\n  {brain}")
        print(f"    Rank stability  (ρ ≥ 0.85)  : {icon_rho} ρ = {v['last_rho']:.4f}")
        print(f"    Slot stability  (≥16/19)    : {icon_ovl} overlap = {v['last_overlap']}/19")
        print(f"    Score variance  (CV ≤ 0.20) : {icon_cv} median CV = {v['median_cv']:.4f}")
        print(f"    ─ {v['verdict']} ─", end="")
        if "MORE NEEDED" in v["verdict"]:
            print(f"  (recommend ≥ {v['recommend_n']} total iterations)")
        else:
            print()
    print()
    if all_sufficient:
        print("  ✅ ALL BRAINS CONVERGED — current iteration count is sufficient.")
    else:
        worst = max(_verdicts.values(), key=lambda v: v["recommend_n"])
        print(f"  ⚠️  NOT ALL BRAINS CONVERGED — run at least "
              f"{worst['recommend_n']} total iterations and re-assess.")
    print("═"*70)
    print(f"\n📂 Audit complete. Full data: {report_path}")
else:
    print("\n⚠️ [BLOCK 9] No data to audit — collection failed in Block 7.")
