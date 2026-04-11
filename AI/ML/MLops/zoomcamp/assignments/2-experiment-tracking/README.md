# MLflow Experiment Tracking - Quick Reference

## Launch the MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

UI available at `http://<your_server_ip>:5000`

## Run the Scripts

```bash
# 1. Preprocess data — downloads and prepares train/val/test splits
python preprocess_data.py --raw_data_path ./data --dest_path ./output

# 2. Train — trains RandomForestRegressor with MLflow autologging
python train.py --data_path ./output

# 3. Hyperparameter optimization — runs 15 Hyperopt trials, logs params and RMSE
python hpo.py --data_path ./output

# 4. Register best model — retrains top 5 on test set, registers the best to model registry
python register_model.py --data_path ./output
```