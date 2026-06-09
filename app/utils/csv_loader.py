import io
import pandas as pd
from fastapi import HTTPException

def load_csv(file):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read raw bytes first, then wrap in BytesIO.
        # Directly passing file.file to pandas fails on Vercel serverless
        # because the stream may not be seekable after upload handling.
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    return df