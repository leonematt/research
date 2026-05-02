"""
ML Zoomcamp Homework 5 — Monitoring (combined script)

Covers Q1–Q3 end-to-end. Q4 is done in the Grafana UI.

Prereqs:
  - source ./env.sh    (sets ML_ZOOMCAMP_API_KEY)
  - docker compose up -d
  - data/ and models/ dirs are created automatically

Run with:
  python homework.py
"""

import datetime
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import psycopg
import requests
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    ColumnQuantileMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_FILE = ""
CURRENT_FILE = ""
REFERENCE_PARQUET = DATA_DIR / "reference.parquet"
MODEL_FILE = MODELS_DIR / "lin_reg.bin"

NUM_FEATURES = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
CAT_FEATURES = ["PULocationID", "DOLocationID"]
TARGET = "duration_min"

# Postgres connection — password sourced from env.sh
PG_PASSWORD = os.environ["ML_ZOOMCAMP_API_KEY"]
PG_DSN_ADMIN = f"host=localhost port=5432 user=postgres password={PG_PASSWORD}"
PG_DSN_TEST = (
    f"host=localhost port=5432 dbname=test user=postgres password={PG_PASSWORD}"
)

CREATE_TABLE_SQL = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    fare_quantile_50 float
)
"""

SEND_TIMEOUT = 10
BEGIN = datetime.datetime(2024, 3, 1, 0, 0)
DAYS_IN_MARCH = 31


# ---------------------------------------------------------------------------
# Step 1: download data
# ---------------------------------------------------------------------------

def download_data(year, month):
    file = f"green_tripdata_{year}-{month:02d}.parquet"
    save_path = f"{DATA_DIR}/{file}"

    if os.path.isfile(save_path):
        print(f"{file} already exists at {save_path}, skipping download")
        return file

    print(f"Downloading {file}")
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    resp = requests.get(url, stream=True)
    with open(save_path, "wb") as handle:
        for data in tqdm(
            resp.iter_content(),
            desc=f"{file}",
            postfix=f"save to {save_path}",
            total=int(resp.headers["Content-Length"]),
        ):
            handle.write(data)
    return file

# ---------------------------------------------------------------------------
# Step 2: Q1 — shape of March 2024 data
# ---------------------------------------------------------------------------

def answer_q1():
    march_data = pd.read_parquet(DATA_DIR / CURRENT_FILE)
    print("=" * 60)
    print(f"Q1 — March 2024 shape: {march_data.shape}")
    print(f"Q1 — rows: {march_data.shape[0]}")
    print("=" * 60)
    return march_data


# ---------------------------------------------------------------------------
# Step 3: train baseline model on Jan 2024, save reference parquet
# ---------------------------------------------------------------------------

def train_and_save_reference():
    if MODEL_FILE.exists() and REFERENCE_PARQUET.exists():
        logging.info("model + reference already exist, skipping training")
        return

    jan_data = pd.read_parquet(DATA_DIR / TRAIN_FILE)
    jan_data["duration_min"] = (
        jan_data.lpep_dropoff_datetime - jan_data.lpep_pickup_datetime
    )
    jan_data.duration_min = jan_data.duration_min.apply(
        lambda td: td.total_seconds() / 60
    )
    jan_data = jan_data[(jan_data.duration_min >= 0) & (jan_data.duration_min <= 60)]
    jan_data = jan_data[
        (jan_data.passenger_count > 0) & (jan_data.passenger_count <= 8)
    ]

    train_data = jan_data[:30000]
    val_data = jan_data[30000:].copy()

    model = LinearRegression()
    model.fit(train_data[NUM_FEATURES + CAT_FEATURES], train_data[TARGET])
    val_data["prediction"] = model.predict(val_data[NUM_FEATURES + CAT_FEATURES])

    with open(MODEL_FILE, "wb") as f:
        joblib.dump(model, f)
    val_data.to_parquet(REFERENCE_PARQUET)
    logging.info("saved %s and %s", MODEL_FILE, REFERENCE_PARQUET)


# ---------------------------------------------------------------------------
# Step 4: build the Evidently report — Q2 metric is added here
# ---------------------------------------------------------------------------

def build_report():
    return Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            # Q2: ColumnQuantileMetric on fare_amount with quantile=0.5
            ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        ]
    )


def column_mapping():
    return ColumnMapping(
        prediction="prediction",
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES,
        target=None,
    )


# ---------------------------------------------------------------------------
# Step 5: Postgres backfill loop — Q3
# ---------------------------------------------------------------------------

def prep_db():
    with psycopg.connect(PG_DSN_ADMIN, autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
    with psycopg.connect(PG_DSN_TEST) as conn:
        conn.execute(CREATE_TABLE_SQL)


def calculate_metrics_postgresql(curr, i, raw_data, reference_data, model, report, mapping):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (BEGIN + datetime.timedelta(i)))
        & (raw_data.lpep_pickup_datetime < (BEGIN + datetime.timedelta(i + 1)))
    ].copy()

    if len(current_data) == 0:
        logging.warning("no data for day %d, skipping", i + 1)
        return

    current_data["prediction"] = model.predict(
        current_data[NUM_FEATURES + CAT_FEATURES].fillna(0)
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=mapping,
    )
    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    fare_quantile_50 = result["metrics"][3]["result"]["current"]["value"]

    curr.execute(
        """insert into dummy_metrics(
            timestamp, prediction_drift, num_drifted_columns,
            share_missing_values, fare_quantile_50
        ) values (%s, %s, %s, %s, %s)""",
        (
            BEGIN + datetime.timedelta(i),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            fare_quantile_50,
        ),
    )


def batch_monitoring_backfill():
    raw_data = pd.read_parquet(DATA_DIR / CURRENT_FILE)
    reference_data = pd.read_parquet(REFERENCE_PARQUET)
    with open(MODEL_FILE, "rb") as f:
        model = joblib.load(f)

    report = build_report()
    mapping = column_mapping()

    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect(PG_DSN_TEST, autocommit=True) as conn:
        for i in range(DAYS_IN_MARCH):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(
                    curr, i, raw_data, reference_data, model, report, mapping
                )
            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent for day %d", i + 1)


# ---------------------------------------------------------------------------
# Step 6: Q3 — query MAX(fare_quantile_50)
# ---------------------------------------------------------------------------

def answer_q3():
    with psycopg.connect(PG_DSN_TEST) as conn:
        result = conn.execute(
            "SELECT MAX(fare_quantile_50) FROM dummy_metrics"
        ).fetchone()
    print("=" * 60)
    print(f"Q3 — max daily 0.5 quantile of fare_amount in March 2024: {result[0]}")
    print("    (pick the closest answer choice: 10 / 12.5 / 14.2 / 14.8)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TRAIN_FILE = download_data(2024, 1)
    CURRENT_FILE = download_data(2024, 3)
    answer_q1()
    train_and_save_reference()
    batch_monitoring_backfill()
    answer_q3()
    print()
    print("Q4 — open Grafana at http://localhost:3000, build a panel for")
    print("     fare_quantile_50, then Save dashboard → Export → Save to file,")
    print("     and drop the JSON into ./dashboards/")


if __name__ == "__main__":
    main()