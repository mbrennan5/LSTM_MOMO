# Sovereign Titan — Momentum Feature Specification
## Final Anchor + Ratio Architecture for 1–3 Day LSTM Predictions

All features below are computed on daily OHLCV bars. Each anchor and ratio
will subsequently be transformed through triple lenses: **Z-score**, **Z-slope**
(rolling slope of Z), and **Z-SOS** (slope of Z-slope). Use rolling windows
appropriate for 1–3 day prediction (typically 10/30/90 day lens windows).

Parameters referenced below:
- `n_fast` = 3–5 days (tune via Optuna)
- `n_mid` = 10–14 days
- `n_slow` = 21 days
- `n_mem` = 63 days (memory/statistical window — keep this longer even for short-term predictions; it provides context)
- `epsilon_rqa` = 0.15 (recurrence threshold as fraction of std)

---

## PART 1: ANCHORS (12 raw features before lenses)

---

### A1. log_roc_fast — Log-Space Velocity
```
log_roc_fast = ln(close / close.shift(n_fast))
```
**Why:** Foundational velocity in log-space. Symmetric for gains/losses,
additive over time. Preferred over arithmetic ROC as the base velocity seed
because log returns have better statistical properties (closer to normal).

---

### A2. roc_fast — Arithmetic Velocity
```
roc_fast = (close - close.shift(n_fast)) / close.shift(n_fast) * 100
```
**Why:** Kept alongside log_roc_fast because the two diverge meaningfully
during large moves (log compresses extremes, arithmetic preserves magnitude).
The LSTM benefits from seeing both representations. Many ratio denominators
reference raw ROC.

---

### A3. geometric_curvature (κ) — Path Shape
```
velocity = (close - close.shift(1)) / 1        # first derivative (1-bar diff)
acceleration = (velocity - velocity.shift(1)) / 1  # second derivative

κ = |acceleration| / (1 + velocity²)^1.5
```
**Why:** Measures the literal "bend" in the price path using differential
geometry. The (1 + v²)^1.5 denominator normalizes for speed — a curve at
high velocity is less significant than the same curve at low velocity.
This nonlinear denominator cannot be replicated by Z-slope of simpler features.

**Implementation note:** Velocity and acceleration here use 1-bar diffs
for maximum responsiveness at 1–3 day horizons.

---

### A4. spectral_signature (PSR) — Frequency Domain Character
```
from scipy.signal import welch

roc_1 = (close - close.shift(1)) / close.shift(1)  # 1-bar ROC stream

def compute_psr(window_of_roc_values):
    freqs, psd = welch(window_of_roc_values, nperseg=min(len(window_of_roc_values), 256))
    low_freq_mask = freqs < (1 / n_mem)   # cycles longer than n_mem days
    return psd[low_freq_mask].sum() / psd.sum()

PSR = roc_1.rolling(n_mem).apply(compute_psr)
```
**Why:** Ratio of low-frequency (trend) power to total power in the
return stream. High PSR = returns dominated by slow trend component.
Low PSR = returns dominated by high-frequency noise/chop.
Completely orthogonal to all time-domain features.

**Implementation note:** Keep the n_mem (63-day) window here even for
short-term predictions. PSR describes the *context* the LSTM operates in,
not the prediction target. Shorter windows produce unstable PSD estimates.

---

### A5. rqa_determinism (DET) — Recurrence Structure
```
roc_1 = (close - close.shift(1)) / close.shift(1)
window = max(25, n_mem)  # minimum 25 bars for meaningful recurrence matrix

def compute_det(x):
    x = np.array(x)
    std_x = np.std(x)
    if std_x == 0:
        return 0.0
    # Build recurrence matrix: R[i,j] = 1 if |x[i] - x[j]| < epsilon * std
    dist = np.abs(x[:, None] - x[None, :])
    R = (dist < epsilon_rqa * std_x).astype(int)
    
    # Count points on diagonal lines of length >= 2
    total_recurrence = np.sum(R) - len(x)  # exclude main diagonal
    if total_recurrence == 0:
        return 0.0
    
    diagonal_points = 0
    for k in range(1, len(x)):
        diag = np.diag(R, k)
        # Count points that are part of runs of length >= 2
        in_run = False
        run_len = 0
        for val in diag:
            if val == 1:
                run_len += 1
            else:
                if run_len >= 2:
                    diagonal_points += run_len
                run_len = 0
                in_run = False
            if run_len >= 2:
                in_run = True
        if run_len >= 2:
            diagonal_points += run_len
    
    return diagonal_points / total_recurrence

DET = roc_1.rolling(window).apply(compute_det)
```
**Why:** Measures what fraction of recurrent states in the return stream
form deterministic (diagonal) patterns vs. isolated (stochastic) recurrences.
High DET = the market is repeating structured patterns.
Low DET = returns are stochastic noise.

**Implementation note:** This is computationally expensive (O(n²) per window).
Consider precomputing and caching. Minimum window of 25 bars required for
the recurrence matrix to be statistically meaningful.

---

### A6. RSI — Relative Strength Index (Wilder)
```
delta = close.diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.ewm(alpha=1/n_mid, min_periods=n_mid).mean()
avg_loss = loss.ewm(alpha=1/n_mid, min_periods=n_mid).mean()

RSI = 100 - (100 / (1 + avg_gain / avg_loss))
```
**Why:** Classic bounded oscillator. Under triple lenses, Z-score of RSI
captures "how extreme is the current overbought/oversold relative to recent
history" and Z-slope captures momentum of sentiment shift. The Wilder
smoothing (exponential) is intentional — CMO provides the unsmoothed
counterpart (see A11).

---

### A7. Chande Trend Meter (CTM) — Multi-Component Trend Score
```
def pct_b(series, n):
    mid = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return ((series - lower) / (upper - lower)) * 10

# Bollinger %B across 4 periods, applied to high/low/close
periods = [20, 50, 75, 100]
columns = ['high', 'low', 'close']
bb_sum = sum(pct_b(df[col], p) for p in periods for col in columns)

# Z-Score component (100-bar)
z_score = ((df['close'] - df['close'].rolling(100).mean())
           / df['close'].rolling(100).std()) * 10

# RSI component (simplified, 14-bar)
rsi_scaled = (df['close'].diff().gt(0).rolling(14).mean() * 100) / 10

# Price channel component (2-bar)
chan_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
price_chan = ((df['close'] - df['low'].rolling(20).min()) / chan_range) * 10

CTM = bb_sum + z_score + rsi_scaled + price_chan
```
**Why:** Composite trend strength that aggregates multiple timeframes and
multiple price dimensions (H/L/C) into a single scalar. Rich internal
structure means Z-slope of CTM captures trend *momentum* across all
its sub-components simultaneously.

---

### A8. Octane Oscillator — Volatility Asymmetry
```
log_ret = ln(close / close.shift(1))

up_returns = log_ret.where(log_ret > 0, 0.0)
dn_returns = log_ret.where(log_ret < 0, 0.0)

up_var = up_returns.rolling(n_slow).var()
dn_var = dn_returns.rolling(n_slow).var()

octane = (up_var - dn_var) / (up_var + dn_var + 1e-10)
```
**Why:** Measures internal price "health" — whether volatility is
concentrated on the upside (bullish conviction) or downside (structural rot).
Two moves with identical log_roc_fast can have opposite Octane readings.
Range: [-1, +1]. High positive = upside dominance. Negative = downside churn.

**Lens synergy:** Z-slope of Octane is an early exhaustion signal — when
Octane's Z-slope turns negative during an uptrend, conviction is fading
before price reverses.

---

### A9. Changepoint Detection Score (ν) — Regime Disequilibrium
```
def cpd_score(price, lookback=n_mem, short_win=n_fast, long_win=n_slow):
    log_ret = np.log(price / price.shift(1))
    
    # Compare short-window statistics to long-window statistics
    mu_short = log_ret.rolling(short_win).mean()
    mu_long = log_ret.rolling(long_win).mean()
    sigma_long = log_ret.rolling(long_win).std()
    
    # Standardized divergence between regimes
    nu = (mu_short - mu_long) / (sigma_long + 1e-10)
    
    return nu
```
**Why:** Quantifies how much the recent (short window) return regime has
diverged from the established (long window) regime. High |ν| = market is
in disequilibrium, a new regime may be forming. Helps the LSTM switch
dynamically between momentum-continuation and mean-reversion strategies.

**Short-term value:** CPD helps exploit (but not overreact to) noise at
1–3 day timescales. The Z-score lens normalizes the score against its own
history, flagging *unusual* changepoints vs. routine fluctuation.

**Note:** This is a simplified version. Full Gaussian Process CPD (e.g.,
Bayesian Online Changepoint Detection / Adams-MacKay) is more principled
but computationally heavy. This approximation captures 80% of the value.

---

### A10. Directional Persistence — Streak Z-Score
```
ret_sign = np.sign(close.diff())

# Count consecutive same-direction closes
groups = (ret_sign != ret_sign.shift()).cumsum()
streaks = ret_sign.groupby(groups).cumcount() + 1

# Z-score the streak length
persistence = (streaks - streaks.rolling(n_mem).mean()) / (streaks.rolling(n_mem).std() + 1e-10)
```
**Why:** Captures "streak psychology" — unusual runs of consecutive
up or down closes that indicate climax behavior. The Z-score normalization
makes it adaptive (a 5-day streak means different things in trending vs.
choppy markets).

**Short-term value:** Extreme Z-scores (|persistence| > 2) identify
high-probability 1-day reactionary bursts.

---

### A11. CMO — Chande Momentum Oscillator (Unsmoothed)
```
delta = close.diff()
sum_up = delta.where(delta > 0, 0.0).rolling(n_mid).sum()
sum_dn = (-delta.where(delta < 0, 0.0)).rolling(n_mid).sum()

CMO = ((sum_up - sum_dn) / (sum_up + sum_dn + 1e-10)) * 100
```
**Why:** Unsmoothed momentum oscillator. Unlike RSI (which uses Wilder
exponential smoothing), CMO applies NO internal smoothing — the triple
lens system provides the smoothing at the appropriate level. This gives
the LSTM the most responsive "source code" for momentum turning points.

**Range:** [-100, +100]. More volatile than RSI by design.

**Relationship to RSI:** RSI and CMO are mathematically related
(CMO = 2 * RSI - 100 in theory), but the smoothing difference makes
them diverge in practice, especially at turning points where CMO leads.

---

### A12. LBR Pinball — RSI of ROC (Fast Oscillator)
```
roc_fast_raw = (close - close.shift(n_fast)) / close.shift(n_fast) * 100

delta = roc_fast_raw.diff()
gain = delta.where(delta > 0, 0.0).rolling(n_fast).mean()
loss = (-delta.where(delta < 0, 0.0)).rolling(n_fast).mean()

rs = gain / (loss + 1e-10)
pinball = 100 - (100 / (1 + rs))
```
**Why:** 3-period RSI applied to 3-period ROC. Purpose-built for 1–3 day
mean reversion by Linda Bradford Raschke. Catches exhaustion in the
*velocity* of price change, not price level.

**Note:** Pinball is both an anchor AND a ratio numerator (Pinball/GARR).
As an anchor under lenses, it provides Z-score of the oscillator itself.
In the ratio, it provides the cross-domain divergence signal.

---

## PART 2: RATIOS (5 cross-domain features before lenses)

Each ratio compares two features from DIFFERENT mathematical domains.
The ratio itself is then passed through the same triple-lens transform.

---

### R1. Curve / Half-Life (κ / MHL) — Geometric Bend vs. Memory Decay
```
# Numerator: geometric curvature (from A3 above)
kappa = geometric_curvature  # see A3

# Denominator: simplified half-life proxy (NOT full ACF-based MHL)
# Use ROC ratio as computationally stable proxy for momentum decay
half_life_proxy = abs(roc_fast) / (abs(raw_roc(close, n_mid)) + 1e-10)
# When fast ROC >> mid ROC, momentum is front-loaded (short half-life)
# When fast ROC ≈ mid ROC, momentum is distributed (long half-life)

ratio_curve_halflife = kappa / (half_life_proxy + 1e-10)
```
**Interpretation:**
- High κ / High half-life proxy (front-loaded) = **"Ballistic Blow-off"**
  Parabolic bend + collapsing persistence. Exit signal.
- High κ / Low half-life proxy (distributed) = **"Controlled Expansion"**
  Accelerating path with staying power. Hold signal.

**Domain cross:** Differential geometry × autocorrelation/memory

---

### R2. GARR / ER — Structural Concentration vs. Path Efficiency
```
# Numerator: GARR (upgraded, from existing feature set)
log_ret = np.log(close / close.shift(1))
GARR = log_ret.rolling(n_fast).sum() / (log_ret.rolling(n_slow).sum() + 1e-10)

# Denominator: Kaufman Efficiency Ratio
net_move = abs(close - close.shift(n_slow))
sum_steps = close.diff().abs().rolling(n_slow).sum()
ER = net_move / (sum_steps + 1e-10)

ratio_garr_er = GARR / (ER + 1e-10)
```
**Interpretation:**
- High GARR / High ER = **"Coherent Thrust"**
  Concentrated momentum on a straight path. Continuation likely.
- High GARR / Low ER = **"Noisy Concentration"**
  Momentum concentrated in bursts but path is choppy. Unreliable.
- Low GARR / High ER = **"Slow Grind"**
  Efficient path but no concentration. Trend intact, low urgency.

**Domain cross:** Log-return concentration × path geometry

---

### R3. Winding / Persistence — Topological Rotation vs. Statistical Streaks
```
# Numerator: Phase-space winding number
price_z = (close - close.rolling(n_slow).mean()) / (close.rolling(n_slow).std() + 1e-10)
roc_z = (roc_fast - roc_fast.rolling(n_slow).mean()) / (roc_fast.rolling(n_slow).std() + 1e-10)

angles = np.arctan2(roc_z, price_z)
angle_diffs = angles.diff()
# Wrap to [-π, π]
angle_diffs = ((angle_diffs + np.pi) % (2 * np.pi)) - np.pi
winding = angle_diffs.rolling(n_slow).sum() / (2 * np.pi)

# Denominator: directional persistence (from A10)
persistence = directional_persistence  # see A10

ratio_winding_persistence = winding / (persistence + 1e-10)
```
**Interpretation:**
- High |winding| / Low persistence = **"Oscillating Regime"**
  Phase space is rotating but no streak dominance. Mean-reversion environment.
- Low |winding| / High persistence = **"Directional Regime"**
  No rotation, strong streaks. Trend-following environment.
- This ratio directly classifies the market's dynamical mode.

**Domain cross:** Topology (phase-space rotation) × order statistics (streaks)

---

### R4. Pinball / GARR — Fast Oscillator vs. Structural Momentum
```
# Numerator: LBR Pinball (from A12)
pinball = lbr_pinball  # see A12

# Denominator: GARR rescaled to comparable range
# GARR is typically [-2, +2], Pinball is [0, 100]
# Rescale GARR to [0, 100] range for clean ratio
garr_scaled = (GARR * 100 + 50).clip(1, 100)  # shift and bound

ratio_pinball_garr = pinball / garr_scaled
```
**Interpretation:**
- Pinball high / GARR low (divergence) = **"Rubber Band"**
  Fast oscillator is overbought but structural momentum is weak.
  Classic 1–3 day mean-reversion setup.
- Pinball high / GARR high (confirmation) = **"Full Send"**
  Both fast and slow momentum aligned. Continuation likely.
- Pinball low / GARR high = **"Pullback in Trend"**
  Short-term weakness within structural strength. Dip-buy candidate.

**Domain cross:** RSI-of-ROC oscillator × log-return concentration

---

### R5. Phase Distance / ER — Multi-Dimensional Energy vs. Path Efficiency
```
# Numerator: Phase-space distance (Euclidean in price-velocity space)
# Normalize both axes to comparable scale first
price_norm = (close - close.rolling(n_slow).mean()) / (close.rolling(n_slow).std() + 1e-10)
roc_norm = (roc_fast - roc_fast.rolling(n_slow).mean()) / (roc_fast.rolling(n_slow).std() + 1e-10)

# Rolling cumulative distance in 2D phase space
dp = price_norm.diff()
dr = roc_norm.diff()
step_distance = np.sqrt(dp**2 + dr**2)
phase_distance = step_distance.rolling(n_slow).sum()

# Denominator: Kaufman ER (same as R2)
ER = net_move / (sum_steps + 1e-10)

ratio_phase_er = phase_distance / (ER + 1e-10)
```
**Interpretation:**
- High distance / High ER = **"Ballistic Efficiency"**
  Huge energy in a straight line. High probability of 1-day extension.
- High distance / Low ER = **"Expensive Energy"**
  Massive 2D movement producing zero net progress. Volatility regime
  flip precursor. The market is "confused" and burning energy.
- Low distance / High ER = **"Quiet Drift"**
  Efficient path with low energy. Calm trend, low vol environment.

**Domain cross:** Phase-space geometry (2D energy) × path efficiency

---

## PART 3: TRIPLE LENS TRANSFORM

Apply to ALL 17 features (12 anchors + 5 ratios):

```python
def apply_triple_lens(feature, windows=[10, 30, 90]):
    """
    For each window w in windows, compute:
      Z     = (feature - rolling_mean(w)) / (rolling_std(w) + 1e-10)
      Z_slope = Z.diff(1)  # or linear regression slope over small sub-window
      Z_SOS   = Z_slope.diff(1)  # slope of slope
    
    Returns 3 columns per window per feature.
    With 3 windows: 9 columns per feature.
    With 17 features: 153 total lens-transformed columns.
    """
    results = {}
    for w in windows:
        z = (feature - feature.rolling(w).mean()) / (feature.rolling(w).std() + 1e-10)
        z_slope = z.diff()       # velocity of Z
        z_sos = z_slope.diff()   # acceleration of Z (slope of slope)
        
        results[f'z_{w}'] = z
        results[f'zslope_{w}'] = z_slope
        results[f'zsos_{w}'] = z_sos
    
    return results
```

**For bounded features (RSI, CMO, Pinball, Octane, Persistence):**
Consider using rolling percentile rank instead of Z-score for the
base lens, since these features have natural bounds and Z-score
can produce misleading values near the boundaries:

```python
def rolling_pct_rank(feature, window):
    return feature.rolling(window).rank(pct=True)
```

---

## SUMMARY TABLE

| # | Feature | Type | Domain | Pillar |
|---|---------|------|--------|--------|
| A1 | log_roc_fast | Anchor | Log velocity | Mismatch seed |
| A2 | roc_fast | Anchor | Arithmetic velocity | Mismatch seed |
| A3 | geometric_curvature | Anchor | Differential geometry | Structure |
| A4 | spectral_signature | Anchor | Frequency domain | Structure |
| A5 | rqa_determinism | Anchor | Recurrence topology | Structure |
| A6 | RSI | Anchor | Bounded oscillator | Mismatch seed |
| A7 | CTM | Anchor | Multi-component trend | Regime |
| A8 | octane | Anchor | Volatility asymmetry | Structure |
| A9 | cpd_score | Anchor | Regime detection | Regime |
| A10 | dir_persistence | Anchor | Order statistics | Regime |
| A11 | CMO | Anchor | Unsmoothed oscillator | Mismatch seed |
| A12 | lbr_pinball | Anchor | RSI-of-ROC oscillator | Mismatch seed |
| R1 | curve_halflife | Ratio | Geometry × Memory | Mismatch+Regime |
| R2 | garr_er | Ratio | Concentration × Efficiency | Mismatch |
| R3 | winding_persistence | Ratio | Topology × Statistics | Regime |
| R4 | pinball_garr | Ratio | Oscillator × Concentration | Mismatch |
| R5 | phase_dist_er | Ratio | Phase-space × Efficiency | Regime |

**Total before lenses:** 17 features
**Total after triple lens (3 windows × 3 transforms):** 153 columns
**Pillar coverage:** Mismatch (7), Structure (4), Regime (6) — all three represented
