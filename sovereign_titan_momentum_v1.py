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
    from scipy.signal import welch as _scipy_welch
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
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
# ### BLOCK 2: NUMBA JIT KERNELS
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
        out[i] = _lin_slope_nb(arr[i - window + 1 : i + 1])
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
    """Wilder RSI — returns array same length as prices."""
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
    """Winding number W = Σ Δθ / 2π in (price_z, roc_z) phase space."""
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
        if i == 0:
            s = 1
        else:
            ps = 1 if returns[i-1] > 0 else (-1 if returns[i-1] < 0 else 0)
            cs = 1 if returns[i]   > 0 else (-1 if returns[i]   < 0 else 0)
            s  = s + 1 if (cs != 0 and cs == ps) else 1
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
        out[i] = _rqa_det_v2_nb(arr[i - window + 1 : i + 1], eps_factor)
    return out


def _warm_up_numba():
    # Skip eager warm-up on CPU — Numba will JIT on first real call instead.
    # Eager compilation hangs in Colab CPU runtimes due to IR analysis overhead.
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚡ Numba warm-up skipped on CPU (kernels compile on first use)")
        return
    try:
        d  = np.random.randn(120).astype(np.float64)
        d2 = np.random.randn(120).astype(np.float64)
        _rolling_linslope(d, 10)
        _rolling_wma(d, 10)
        _kalman_numba(d)
        _rsi_nb(d, 14)
        _rolling_geom_curv(d, d2)
        _rolling_phase_winding(d, d2, 20)
        _streak_zscore(d, 30)
        _rolling_rqa_det_v2(d, 25)
        print("✅ Numba momentum kernels compiled and ready")
    except Exception as e:
        print(f"⚠️  Numba warm-up skipped: {e}")

_warm_up_numba()


# ==============================================================================
# ### BLOCK 3: MOMENTUM FEATURE FACTORY  v2.0
#
# Architecture: 12 Anchors + 5 Cross-Domain Ratios = 17 seeds
# Each seed → triple lens [10, 30, 90] × {z, z_slope, z_sos} = 153 columns
#
# Parameters:
#   n_fast = 3   |  n_mid = 14  |  n_slow = 21  |  n_mem = 63
#   epsilon_rqa  = 0.15
#   LENS_WINDOWS = [10, 30, 90]
#   BOUNDED      = {rsi, cmo, lbr_pinball, octane, dir_persist}
#                  → rolling pct-rank replaces Z-score for these
#
# Pillars:  Mismatch (7)  |  Structure (4)  |  Regime (6)
# ==============================================================================

def _psr_welch(roc_arr: np.ndarray, window: int, f_low_period: int) -> np.ndarray:
    """
    Power Spectral Ratio via Welch PSD.
    Falls back to numpy FFT if scipy is unavailable.
    Returns fraction of power at f <= 1/f_low_period.
    """
    n   = len(roc_arr)
    out = np.full(n, np.nan)
    f_low = 1.0 / f_low_period
    for i in range(window - 1, n):
        seg = roc_arr[i - window + 1 : i + 1].copy()
        seg -= seg.mean()
        if np.std(seg) < 1e-10:
            out[i] = 0.5
            continue
        if _HAS_SCIPY:
            f, psd = _scipy_welch(seg, nperseg=len(seg), noverlap=0)
        else:
            psd = np.abs(np.fft.rfft(seg)) ** 2
            f   = np.fft.rfftfreq(len(seg))
        total  = psd.sum()
        out[i] = psd[f <= f_low].sum() / total if total > 1e-12 else 0.5
    return out


def generate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

    # ── Parameters ────────────────────────────────────────────────────────────
    n_fast       = 3
    n_mid        = 14
    n_slow       = 21
    n_mem        = 63
    epsilon_rqa  = 0.15
    LENS_WINDOWS = [10, 30, 90]
    BOUNDED      = {'rsi', 'cmo', 'lbr_pinball', 'octane', 'dir_persist'}

    # ==========================================================================
    # ANCHORS
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
    psr   = np.nan_to_num(
                _psr_welch(roc_1, window=n_mem, f_low_period=n_mem),
                nan=0.5)

    # A5. RQA Determinism DET  (diagonal lines ≥ 2, window = max(25, n_mem))
    window_rqa = max(25, n_mem)
    rqa_det    = _rolling_rqa_det_v2(roc_1, window_rqa, epsilon_rqa)

    # A6. RSI — Wilder (n_mid period)
    delta    = cl_s.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/n_mid, min_periods=n_mid).mean()
    avg_loss = loss.ewm(alpha=1.0/n_mid, min_periods=n_mid).mean()
    rsi      = (100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))).values

    # A7. Chande Trend Meter (CTM)
    def _pct_b(series, period):
        mid   = series.rolling(period).mean()
        std   = series.rolling(period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        return ((series - lower) / (upper - lower + 1e-9)) * 10

    ctm_raw = np.zeros(n)
    for p in [20, 50, 75, 100]:
        for s in [cl_s, hi_s, lo_s]:
            ctm_raw += _pct_b(s, p).values
    ctm_raw += ((cl_s - cl_s.rolling(100).mean()) /
                (cl_s.rolling(100).std() + 1e-9) * 10).values
    ctm_raw += (cl_s.diff().gt(0).rolling(14).mean() * 100 / 10).values
    chan_range = hi_s.rolling(20).max() - lo_s.rolling(20).min()
    ctm_raw  += ((cl_s - lo_s.rolling(20).min()) /
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

    # A10. Directional Persistence — streak z-score  (n_mem window)
    daily_ret   = np.diff(cl, prepend=cl[0]) / (cl + 1e-9)
    dir_persist = _streak_zscore(daily_ret, n_mem)

    # A11. CMO — unsmoothed Chande Momentum Oscillator  [-100, +100]
    delta_s = cl_s.diff().fillna(0)
    sum_up  = delta_s.where(delta_s > 0, 0.0).rolling(n_mid).sum().values
    sum_dn  = (-delta_s.where(delta_s < 0, 0.0)).rolling(n_mid).sum().values
    cmo     = ((sum_up - sum_dn) / (sum_up + sum_dn + 1e-10)) * 100

    # A12. LBR Pinball — RSI (rolling mean) of ROC_fast  [0, 100]
    roc_fast_raw = cl_s.pct_change(n_fast).fillna(0) * 100
    d_pb  = roc_fast_raw.diff()
    g_pb  = d_pb.where(d_pb > 0, 0.0).rolling(n_fast).mean()
    l_pb  = (-d_pb.where(d_pb < 0, 0.0)).rolling(n_fast).mean()
    lbr_pinball = (100 - (100 / (1 + g_pb / (l_pb + 1e-9)))).values

    # ==========================================================================
    # SHARED INTERMEDIATES  (used by multiple ratios)
    # ==========================================================================

    # GARR: log-return concentration  (n_fast rolling / n_slow rolling sum)
    garr = (log_ret.rolling(n_fast).sum() /
            (log_ret.rolling(n_slow).sum() + 1e-10)).values

    # Kaufman Efficiency Ratio  (n_slow window)
    net_move  = (cl_s - cl_s.shift(n_slow)).abs()
    sum_steps = cl_s.diff().abs().rolling(n_slow).sum()
    er        = (net_move / (sum_steps + 1e-10)).values

    # Phase-space coordinates  (normalised to n_slow rolling z)
    roc_fast_s    = pd.Series(roc_fast, index=idx)
    price_z_nslow = ((cl_s - cl_s.rolling(n_slow).mean()) /
                     (cl_s.rolling(n_slow).std() + 1e-10)).values.astype(np.float64)
    roc_z_nslow   = ((roc_fast_s - roc_fast_s.rolling(n_slow).mean()) /
                     (roc_fast_s.rolling(n_slow).std() + 1e-10)).values.astype(np.float64)

    # ==========================================================================
    # RATIOS
    # ==========================================================================

    # R1. Curve / Half-Life  (geometry × memory)
    roc_mid_arr     = cl_s.pct_change(n_mid).fillna(0).values * 100
    half_life_proxy = np.abs(roc_fast) / (np.abs(roc_mid_arr) + 1e-10)
    curve_halflife  = geom_curv / (half_life_proxy + 1e-10)

    # R2. GARR / ER  (log-return concentration × path efficiency)
    garr_er = garr / (er + 1e-10)

    # R3. Winding / Persistence  (topology × order statistics)
    phase_wind          = _rolling_phase_winding(price_z_nslow, roc_z_nslow,
                                                 window=n_slow)
    winding_persistence = phase_wind / (dir_persist + 1e-10)

    # R4. Pinball / GARR  (fast oscillator × structural momentum)
    garr_scaled  = np.clip(garr * 100 + 50, 1.0, 100.0)
    pinball_garr = lbr_pinball / (garr_scaled + 1e-10)

    # R5. Phase Distance / ER  (2-D energy × path efficiency)
    dp            = np.diff(price_z_nslow, prepend=price_z_nslow[0])
    dr            = np.diff(roc_z_nslow,   prepend=roc_z_nslow[0])
    step_dist     = np.sqrt(dp**2 + dr**2)
    phase_dist    = pd.Series(step_dist, index=idx).rolling(n_slow).sum().values
    phase_dist_er = phase_dist / (er + 1e-10)

    # ==========================================================================
    # SEED DICT  (12 anchors + 5 ratios = 17 seeds)
    # ==========================================================================
    ALL_SEEDS: dict = {
        'log_roc_fast':        pd.Series(log_roc_fast,        index=idx),
        'roc_fast':            pd.Series(roc_fast,            index=idx),
        'geom_curv':           pd.Series(geom_curv,           index=idx),
        'psr':                 pd.Series(psr,                 index=idx),
        'rqa_det':             pd.Series(rqa_det,             index=idx),
        'rsi':                 pd.Series(rsi,                 index=idx),
        'ctm':                 pd.Series(ctm,                 index=idx),
        'octane':              pd.Series(octane,              index=idx),
        'cpd_score':           pd.Series(cpd_score,           index=idx),
        'dir_persist':         pd.Series(dir_persist,         index=idx),
        'cmo':                 pd.Series(cmo,                 index=idx),
        'lbr_pinball':         pd.Series(lbr_pinball,         index=idx),
        'curve_halflife':      pd.Series(curve_halflife,      index=idx),
        'garr_er':             pd.Series(garr_er,             index=idx),
        'winding_persistence': pd.Series(winding_persistence, index=idx),
        'pinball_garr':        pd.Series(pinball_garr,        index=idx),
        'phase_dist_er':       pd.Series(phase_dist_er,       index=idx),
    }

    # ==========================================================================
    # TRIPLE LENS TRANSFORM
    # 17 seeds × 3 windows × 3 transforms = 153 LENS columns
    # Bounded features → rolling pct-rank as Z base
    # z_slope = z.diff()   |   z_sos = z_slope.diff()
    # ==========================================================================
    for name, series in ALL_SEEDS.items():
        arr   = series.values.astype(np.float64)
        arr_s = pd.Series(arr, index=idx)
        for w in LENS_WINDOWS:
            if name in BOUNDED:
                z = arr_s.rolling(w).rank(pct=True).values
            else:
                rm = arr_s.rolling(w).mean().values
                rs = arr_s.rolling(w).std().values
                z  = (arr - rm) / (rs + 1e-9)
            z_s     = pd.Series(z, index=idx)
            z_slope = z_s.diff().values
            z_sos   = pd.Series(z_slope, index=idx).diff().values
            df[f'LENS_{w}_{name}_z']       = z
            df[f'LENS_{w}_{name}_z_slope'] = z_slope
            df[f'LENS_{w}_{name}_z_sos']   = z_sos

    return (df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .dropna(subset=['T_FINAL'])
              .fillna(0))


# ==============================================================================
# ### BLOCK 4: PARALLEL LOADER
# ==============================================================================
def fetch_data(symbol):
    import time
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            data   = ticker.history(period="3y", interval="1d", auto_adjust=True)
            if data is None or data.empty:
                return None
            data.columns = [str(c).lower() for c in data.columns]
            return data if 'close' in data.columns else None
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None

def load_hybrid_data_parallel(brain_name, symbol_list, dl_workers=8):
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
                       seq_len=60, epochs=50, batch_size=2048):
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
