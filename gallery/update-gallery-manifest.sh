#!/bin/bash

echo "Generating gallery manifest..."

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

MANIFEST_FILE="$PROJECT_ROOT/gallery/data/gallery-manifest.json"
mkdir -p "$(dirname "$MANIFEST_FILE")"

echo "{" > "$MANIFEST_FILE"
echo "  \"lastUpdated\": \"$(date +"%Y-%m-%d %H:%M:%S")\"," >> "$MANIFEST_FILE"

echo "  \"contentImages\": [" >> "$MANIFEST_FILE"
find "$PROJECT_ROOT/model/assets/input" -type f \( -name "*.jpg" -o -name "*.png" \) | sort | while read -r file; do
  rel_path="${file#$PROJECT_ROOT/}"
  echo "    \"/$rel_path\"," >> "$MANIFEST_FILE"
done
sed -i '$ s/,$//' "$MANIFEST_FILE"
echo "  ]," >> "$MANIFEST_FILE"

echo "  \"styleImages\": [" >> "$MANIFEST_FILE"
find "$PROJECT_ROOT/model/assets/reference" -type f \( -name "*.jpg" -o -name "*.png" \) | sort | while read -r file; do
  rel_path="${file#$PROJECT_ROOT/}"
  echo "    \"/$rel_path\"," >> "$MANIFEST_FILE"
done
sed -i '$ s/,$//' "$MANIFEST_FILE"
echo "  ]," >> "$MANIFEST_FILE"

echo "  \"presetResults\": [" >> "$MANIFEST_FILE"
mkdir -p "$PROJECT_ROOT/model/assets/_results"
find "$PROJECT_ROOT/model/assets/_results" -type f \( -name "*.jpg" -o -name "*.png" \) | sort | while read -r file; do
  rel_path="${file#$PROJECT_ROOT/}"
  echo "    \"/$rel_path\"," >> "$MANIFEST_FILE"
done
sed -i '$ s/,$//' "$MANIFEST_FILE"
echo "  ]," >> "$MANIFEST_FILE"

echo "  \"customResults\": [" >> "$MANIFEST_FILE"
mkdir -p "$PROJECT_ROOT/model/results"
find "$PROJECT_ROOT/model/results" -type f \( -name "*.jpg" -o -name "*.png" \) | sort | while read -r file; do
  rel_path="${file#$PROJECT_ROOT/}"
  echo "    \"/$rel_path\"," >> "$MANIFEST_FILE"
done
sed -i '$ s/,$//' "$MANIFEST_FILE"
echo "  ]," >> "$MANIFEST_FILE"

echo "  \"bestResults\": [" >> "$MANIFEST_FILE"
mkdir -p "$PROJECT_ROOT/gallery/assets/best"
find "$PROJECT_ROOT/gallery/assets/best" -type f \( -name "*.jpg" -o -name "*.png" \) | sort | while read -r file; do
  rel_path="${file#$PROJECT_ROOT/}"
  echo "    \"/$rel_path\"," >> "$MANIFEST_FILE"
done
sed -i '$ s/,$//' "$MANIFEST_FILE"
echo "  ]" >> "$MANIFEST_FILE"

echo "}" >> "$MANIFEST_FILE"

echo "Manifest generated successfully at: $MANIFEST_FILE"