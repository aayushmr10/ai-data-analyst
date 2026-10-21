"""
pandas/numpy produce values (NaN, np.int64, Timestamp, inf) that the
standard json encoder chokes on. Centralize the cleanup here so every
module can just call safe_json(obj) before returning it from an endpoint.
"""
from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def safe_json(obj):
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray,)):
        return [safe_json(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Series,)):
        return [safe_json(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return safe_json(obj.to_dict(orient="records"))
    return obj
