#!/usr/bin/env bash
# setup_notebooklm.sh — One-time setup for the NotebookLM skill
# Run from the LSTM_MOMO repository root: bash setup_notebooklm.sh

set -e

SKILL_DIR="$HOME/.claude/skills/notebooklm"
SKILL_REPO="https://github.com/PleasePrompto/notebooklm-skill"
NOTEBOOK_ID="4ed28b19-7ba3-44b7-a1f9-cba4fa452fdf"
NOTEBOOK_URL="https://notebooklm.google.com/notebook/${NOTEBOOK_ID}"

echo "=== NotebookLM Skill Setup for LSTM_MOMO ==="
echo ""

# --- Prerequisites check ---
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' is required but not found. $2"
        exit 1
    fi
}

check_command git   "Install Git: https://git-scm.com/downloads"
check_command python3 "Install Python 3.10+: https://python.org/downloads"

# Check Python version >= 3.10
PYVER=$(python3 -c "import sys; print(sys.version_info >= (3,10))")
if [ "$PYVER" != "True" ]; then
    echo "ERROR: Python 3.10+ required. Current: $(python3 --version)"
    exit 1
fi

# Chrome check (warn but don't fail — user may install later)
if ! command -v google-chrome &>/dev/null && ! command -v google-chrome-stable &>/dev/null; then
    echo "WARNING: Google Chrome not found."
    echo "         The NotebookLM skill requires Chrome for Google auth reliability."
    echo "         Install from: https://www.google.com/chrome/"
    echo ""
    read -r -p "Continue setup anyway? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 0
fi

# --- Clone or update the skill ---
mkdir -p "$HOME/.claude/skills"

if [ -d "$SKILL_DIR/.git" ]; then
    echo "Skill already cloned. Pulling latest..."
    git -C "$SKILL_DIR" pull --ff-only
else
    echo "Cloning NotebookLM skill..."
    git clone "$SKILL_REPO" "$SKILL_DIR"
fi

echo ""

# --- Copy project notebook config ---
ENV_SRC="$(dirname "$0")/.notebooklm.env"
ENV_DEST="$SKILL_DIR/.env"

if [ -f "$ENV_SRC" ]; then
    if [ -f "$ENV_DEST" ]; then
        echo "Skill .env already exists — skipping overwrite."
        echo "To reset: cp '$ENV_SRC' '$ENV_DEST'"
    else
        cp "$ENV_SRC" "$ENV_DEST"
        echo "Copied notebook config to skill directory."
    fi
fi

echo ""

# --- Add trading notebook to library ---
echo "Adding trading notebook to library..."
if python3 "$SKILL_DIR/scripts/run.py" notebook_manager.py list 2>/dev/null | grep -q "$NOTEBOOK_ID"; then
    echo "Notebook already in library."
else
    python3 "$SKILL_DIR/scripts/run.py" notebook_manager.py add \
        --url "$NOTEBOOK_URL" \
        --name "Trading Studies Library" \
        --description "Ken Long / Tortoise Capital trading methodology, RL10/PSAR systems, momentum indicators, regime classification, backtesting strategies" \
        --topics "trading,ken-long,rl10,psar,momentum,backtesting,regime" \
        && echo "Notebook added successfully." \
        || echo "WARNING: Could not add notebook automatically. Run auth setup first, then re-run this script."
fi

echo ""

# --- Auth setup ---
echo "Checking authentication..."
AUTH_STATUS=$(python3 "$SKILL_DIR/scripts/run.py" auth_manager.py status 2>/dev/null || echo "unauthenticated")

if echo "$AUTH_STATUS" | grep -qi "authenticated\|valid"; then
    echo "Already authenticated with Google."
else
    echo "Authentication required."
    echo ""
    echo "A Chrome window will open. Log in to your Google account."
    echo "Tip: Use a dedicated Google account rather than your primary account."
    echo ""
    read -r -p "Press Enter to open the auth window..."
    python3 "$SKILL_DIR/scripts/run.py" auth_manager.py setup
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Try it in Claude Code:"
echo "  Ask my NotebookLM: What are the key rules for the RL10/PSAR Dragon entry system?"
echo ""
echo "Notebook URL: $NOTEBOOK_URL"
