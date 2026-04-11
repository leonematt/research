#!/usr/bin/env bash
set -euo pipefail

# Create directories
mkdir -p ./data ./output

# Download Green Taxi Trip Records for Jan, Feb, Mar 2023
for month in 01 02 03; do
  url="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-${month}.parquet"
  echo "Downloading ${url}..."
  wget -q -P ./data "$url"
done

echo "Downloaded files:"
ls -lh ./data/

# Run preprocessing
python preprocess_data.py --raw_data_path ./data --dest_path ./output

echo ""
echo "Files saved to ./output:"
ls -lh ./output/
echo ""
echo "Total files: $(ls -1 ./output | wc -l)"
