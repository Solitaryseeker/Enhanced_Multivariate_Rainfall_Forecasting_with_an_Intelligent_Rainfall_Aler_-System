# Yemen Rainfall Forecasting (iTransformer only): local bundle

## Folder contents
index.html (website) | server.py (API + iTransformer loader) | history.csv, pcode_map.json, scalers.json (from your CSV)
prepare_artifacts.py (rebuilds those three) | requirements.txt | run.bat / run.sh | model_config.example.json

## You must add 2 things
1. Your trained `.pth` file in this folder (any file name ending .pth).
2. `model_config.json` containing `{"n_heads": N}`: copy model_config.example.json and use the n_heads of the run that made your .pth. It is not stored in the checkpoint.
Optional but best: `x_scaler.pkl` and `y_scaler.pkl` from training (they override scalers.json).

## Run
    pip install -r requirements.txt
    python server.py          (Windows: double-click run.bat)
Open http://localhost:8000. The terminal prints "Loaded iTransformer: {...}" when the model is ready.
Missing files are listed on the Prediction page; add them and submit again (no restart).

## API contract v1.0
GET /api/health -> {contract_version, model, seq_len, model_loaded, missing[], error}
POST /api/predict {date, pcode, rfh, r1h, r3h} -> {predicted_mm, unit, target_date, window}
errors -> {code: VALIDATION_ERROR|MODEL_NOT_LOADED|INFERENCE_ERROR, error} (HTTP 400/503/500)

## Rules preserved from training
Feature order: rfh, r1h, r3h, rfh_lag_1/2/3/6/12, month_sin, month_cos, dekad_sin, dekad_cos, month, dekad (dekad = 1, 11, 21).
log1p on rfh/r1h/r3h only (lags raw); MinMax scaling; output inverse-scaled then expm1 -> mm. Window 24 dekads; 35 prior dekads needed.
The forecast is for the dekad AFTER the observation date.
