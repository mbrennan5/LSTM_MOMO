# LSTM_MOMO — Claude Code Project Guide

## Project Overview

This repository contains the **Sovereign Titan** LSTM momentum trading system — a feature engineering framework for 1–3 day price prediction using:

- **12 Anchors**: Raw features spanning log velocity, geometric curvature, spectral signatures, recurrence topology, oscillators, and regime detection
- **5 Cross-domain Ratios**: Paired features from different mathematical domains (geometry × memory, concentration × efficiency, topology × statistics, etc.)
- **Triple Lens Transform**: Every feature expanded through Z-score, Z-slope, and Z-SOS across three rolling windows → 153 total LSTM input columns

Key parameters: `n_fast=3–5`, `n_mid=10–14`, `n_slow=21`, `n_mem=63`, `epsilon_rqa=0.15`

---

## NotebookLM Integration

This project is paired with a **NotebookLM trading notebook** containing Ken Long / Tortoise Capital methodology, RL10/PSAR systems, momentum indicators, regime classification, and backtesting strategies.

**Notebook URL:** `https://notebooklm.google.com/notebook/4ed28b19-7ba3-44b7-a1f9-cba4fa452fdf`

### Setup (one-time)

Run the setup script from the repo root:

```bash
bash setup_notebooklm.sh
```

This will:
1. Create `~/.claude/skills/` if it doesn't exist
2. Clone the NotebookLM skill
3. Copy the notebook configuration
4. Prompt you through Google auth

### Using the NotebookLM Skill

Once set up, query the trading notebook directly from Claude Code:

```
Ask my NotebookLM: What are the key rules for the RL10/PSAR Dragon entry system?
```

```
Check my trading notebook for Ken Long's recommended position sizing
```

```
Query NotebookLM about the belly slow metric calculation and trend threshold values
```

```
Ask NotebookLM: How does the regime classification framework map to the Sovereign Titan feature pillars?
```

### Auth Management

```
Check NotebookLM authentication status
Set up NotebookLM authentication
Re-authenticate NotebookLM
```

---

## Key Files

| File | Purpose |
|------|---------|
| `sovereign_titan_momentum_v1.py` | Main feature engineering implementation |
| `momentum_lookback_optimizer.py` | Optuna-based hyperparameter tuning for `n_fast`, `n_mid`, etc. |
| `README.md` | Full feature specification with math and rationale |
| `setup_notebooklm.sh` | One-time NotebookLM skill setup script |
| `.notebooklm.env` | Notebook ID and browser configuration |

---

## Development Notes

- All features must pass through the **triple lens transform** (`apply_triple_lens`) before being fed to the LSTM
- Bounded features (RSI, CMO, Pinball, Octane, Persistence) should use `rolling_pct_rank` for the base Z lens
- The `rqa_determinism` computation is O(n²) per window — precompute and cache in production
- Optuna trials should tune `n_fast` (3–5), `n_mid` (10–14) while keeping `n_mem=63` fixed for PSR/RQA stability
