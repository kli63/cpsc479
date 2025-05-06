#!/bin/bash

echo "Starting the Style Transfer Gallery server..."

echo "Updating gallery manifest..."
./gallery/update-gallery-manifest.sh

if command -v python3 &> /dev/null; then
    echo "Starting server with Python 3..."
    python3 -m http.server 8000
    exit 0
fi

if command -v python &> /dev/null; then
    echo "Starting server with Python..."
    python -m http.server 8000
    exit 0
fi

if command -v npm &> /dev/null; then
    if ! command -v http-server &> /dev/null; then
        echo "Installing http-server..."
        npm install -g http-server
    fi
    echo "Starting server with http-server..."
    http-server -p 8000
    exit 0
fi

echo "Error: Neither Python nor npm was found. Please install one of them to run the server."
exit 1