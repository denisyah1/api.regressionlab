import io
import pandas as pd
from fastapi import HTTPException

def load_csv(file):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = file.file.read()
        # Try common encodings in order — Excel CSVs are often latin-1 or cp1252
        for encoding in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise HTTPException(status_code=400, detail="Could not decode CSV file. Please save it as UTF-8.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    return df