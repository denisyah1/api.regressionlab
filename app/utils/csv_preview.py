import io
import pandas as pd
from fastapi import HTTPException
from app.utils.json_sanitizer import sanitize

def analyze_csv(file, preview_rows: int = 5):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    columns = list(df.columns)

    numeric_columns = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    null_info = {
        col: int(df[col].isnull().sum())
        for col in df.columns
    }

    preview_data = df.head(preview_rows).to_dict(orient="records")

    response = {
        "filename": file.filename,
        "total_rows": len(df),
        "total_columns": len(columns),
        "columns": columns,
        "numeric_columns": numeric_columns,
        "null_summary": null_info,
        "preview": preview_data
    }

    return sanitize(response)