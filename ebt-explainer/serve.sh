#!/usr/bin/env sh
# Serve the explainer locally (the page also works by opening index.html directly).
cd "$(dirname "$0")" && echo "Open http://localhost:${1:-8000}/" && python3 -m http.server "${1:-8000}"
