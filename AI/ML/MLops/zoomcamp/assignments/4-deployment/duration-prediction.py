import sys
import pickle
import pandas as pd
import numpy as np


categorical = ['PULocationID', 'DOLocationID']


def load_model(path='model.bin'):
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df


def main(year, month):
    dv, model = load_model()
    df = read_data(year, month)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'mean predicted duration: {y_pred.mean()}')
    print(f'std  predicted duration: {y_pred.std()}')

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred,
    })
    output_file = f'output_{year:04d}_{month:02d}.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)