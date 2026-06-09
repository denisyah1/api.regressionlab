from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import os
import io
import pickle
from app.utils.csv_preview import analyze_csv
from app.services.regression_service import run_regression, run_regression_and_get_model
from app.utils.csv_loader import load_csv
from app.utils.eda_analyzer import analyze_eda
from fastapi.middleware.cors import CORSMiddleware
from app.utils.recommendation_engine import recommend_regression_columns

app = FastAPI(title="Regression Visualization API")

cors_origins_env = os.getenv("CORS_ORIGINS", "")
cors_origins = [
    origin.strip()
    for origin in cors_origins_env.split(",")
    if origin.strip()
]
if not cors_origins:
    cors_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CSV PREVIEW
# =========================
@app.post("/api/csv/preview")
async def preview_csv(
    file: UploadFile = File(...)
):
    return analyze_csv(file)

# =========================
# REGRESSION (CSV RAW → ML)
# =========================
@app.post("/api/regression")
async def regression(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    feature_columns: str = Form(...),
    null_strategy: str = Form("auto")
):
    features = [c.strip() for c in feature_columns.split(",") if c.strip()]
    return run_regression(
        file=file,
        target_column=target_column,
        feature_columns=features,
        null_strategy=null_strategy
    )

# NOTE: GET /api/regression/plot has been removed.
# Plot data is now returned directly inside POST /api/regression
# under the "plot_data" key. Update your frontend to read it from there.

# =========================
# DOWNLOAD SAVED MODEL
# =========================
# Changed from GET to POST — the model is no longer saved to disk.
# We re-run regression with the same CSV and stream the result directly.
@app.post("/api/model/download")
async def download_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    feature_columns: str = Form(...),
    null_strategy: str = Form("auto")
):
    features = [c.strip() for c in feature_columns.split(",") if c.strip()]
    model, model_name = run_regression_and_get_model(
        file=file,
        target_column=target_column,
        feature_columns=features,
        null_strategy=null_strategy
    )
    # Serialize model into memory buffer — no disk writes at all
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={model_name}.pkl"}
    )

# =========================
# EDA ANALYSIS
# =========================
@app.post("/api/csv/eda")
async def csv_eda(file: UploadFile = File(...)):
    df = load_csv(file)
    return analyze_eda(df)

# ==========================
# REGRESSION RECOMMENDATION
# ==========================
@app.post("/api/csv/recommendation")
async def csv_recommendation(file: UploadFile = File(...)):
    df = load_csv(file)
    return recommend_regression_columns(df)