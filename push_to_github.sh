#!/usr/bin/env bash
# Push this package to the existing public GitHub repository.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

REPO_URL="${1:-https://github.com/gaick/Separation-of-Shallow-and-Deep-Groundwater-Storage-Variations-in-the-Beijing-Tianjin-Hebei-Region.git}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh not found; using git only"
fi

if gh auth status >/dev/null 2>&1; then
  echo "gh authenticated"
else
  echo "NOTE: gh is not logged in. git push may prompt for credentials."
  echo "If push fails, run:  gh auth login --hostname github.com --git-protocol https --web"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git add -A
if ! git diff --cached --quiet || ! git diff --quiet; then
  git -c user.name="gaick" -c user.email="gaick@users.noreply.github.com" commit -m "$(cat <<'EOF'
Add full M2 code package, CAGEO-ready README, and rigorous quick test.

Quick test verifies Persistence residual identity, training, and metrics on synthetic AR(1) fields.
EOF
)" || true
fi

# Prefer authenticated HTTPS via gh if available
if gh auth status >/dev/null 2>&1; then
  gh auth setup-git
fi

git branch -M main
git push -u origin main
echo "✅ Pushed to $REPO_URL"
