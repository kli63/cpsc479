#!/bin/bash

echo "Starting the Style Transfer Gallery server..."

# Check if python3 is available
if command -v python3 &> /dev/null; then
    echo "Starting server with Python 3..."
    cd "$(dirname "$0")/.."
    python3 -m http.server
    exit 0
fi

# Check if python (could be Python 2 or 3) is available
if command -v python &> /dev/null; then
    echo "Starting server with Python..."
    cd "$(dirname "$0")/.."
    python -m http.server
    exit 0
fi

# Check if npm and http-server are available
if command -v npm &> /dev/null; then
    if ! command -v http-server &> /dev/null; then
        echo "Installing http-server..."
        npm install -g http-server
    fi
    echo "Starting server with http-server..."
    cd "$(dirname "$0")/.."
    http-server
    exit 0
fi

echo "Error: Neither Python nor npm was found. Please install one of them to run the server."
exit 1