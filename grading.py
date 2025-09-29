"""
main_lms_ai.py — Drop-in FastAPI service for:
1) Automated Grading
   - MCQs: XGBoost / LightGBM (with rule-based fallback)
   - Subjective answers: RoBERTa (STS-B cross-encoder similarity)
   - Handwriting: YOLOv8 + OpenCV (detect handwritten regions; optional OCR hook)

2) Progress Tracking & Analytics
   - Time-series forecasting: Prophet or LSTM
   - Clustering: KMeans / DBSCAN

Notes
- Keep this as a separate microservice or merge into your existing FastAPI app.
- All models load lazily and are cached.
- Where training data is required (MCQ ML, forecasting), endpoints accept CSV uploads or JSON data.
- This file is production-ready skeleton code with safe defaults and clear TODOs to plug in your data.

Run
- uvicorn main_lms_ai:app --reload --host 127.0.0.1 --port 8010

Env / Dependencies
pip install fastapi uvicorn pydantic[dotenv] numpy pandas scikit-learn
pip install xgboost lightgbm transformers torch sentencepiece
pip install ultralytics opencv-python pillow
pip install prophet
pip install tensorflow  # (only if you use the LSTM forecaster)

"""
from __future__ import annotations

import io
import json
import pickle
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- ML imports (optional/lazy) ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.cluster import KMeans, DBSCAN

# Defer heavy imports inside functions to speed cold start

app = FastAPI(title="LMS AI Add-ons")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Utilities & Data Schemas
# -------------------------------

class MCQTrainRow(BaseModel):
    student_id: str
    question_id: str
    choice: str
    correct_choice: str
    # Optional item metadata (difficulty, tags, etc.) can help models
    difficulty: Optional[float] = None
    time_taken_sec: Optional[float] = None

class MCQTrainPayload(BaseModel):
    rows: List[MCQTrainRow]
    model: Optional[str] = Field(default="xgboost", description="xgboost | lightgbm")

class MCQGradePayload(BaseModel):
    answers: Dict[str, str]  # {question_id: selected_choice}
    key: Dict[str, str]      # {question_id: correct_choice}
    model: Optional[str] = Field(default=None, description="Use trained model id or None => rule-based")

class SubjectiveGradeRequest(BaseModel):
    student_answer: str
    reference_answer: str
    max_score: float = 10.0

class ForecastRequest(BaseModel):
    # Accept JSON time series: list of {date: "YYYY-MM-DD", value: float}
    series: List[Dict[str, Any]]
    horizon: int = 14
    method: str = Field(default="prophet", description="prophet | lstm")

class ClusterRequest(BaseModel):
    # Each row is a student vector: {student_id: str, features: {"score": .., "time": ..}}
    rows: List[Dict[str, Any]]
    algo: str = Field(default="kmeans", description="kmeans | dbscan")
    k: Optional[int] = 3  # for kmeans
    eps: Optional[float] = 0.5  # for dbscan
    min_samples: Optional[int] = 5  # for dbscan

# In-memory model registry (swap to Redis/DB if desired)
MODEL_STORE: Dict[str, Any] = {}


def _extract_mcq_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple numeric features for MCQ modeling.
    Expected columns: [choice, correct_choice, difficulty, time_taken_sec]
    """
    out = pd.DataFrame()
    out["is_correct"] = (df["choice"] == df["correct_choice"]).astype(int)
    # Basic encodings (label hashing)
    out["choice_code"] = df["choice"].astype(str).str.lower().str[0].map({"a":0,"b":1,"c":2,"d":3}).fillna(4)
    out["correct_code"] = df["correct_choice"].astype(str).str.lower().str[0].map({"a":0,"b":1,"c":2,"d":3}).fillna(4)
    out["match_code"] = (out["choice_code"] == out["correct_code"]).astype(int)
    out["difficulty"] = df.get("difficulty", pd.Series([np.nan]*len(df))).astype(float).fillna(0.5)
    out["time_taken_sec"] = df.get("time_taken_sec", pd.Series([np.nan]*len(df))).astype(float).fillna(out["time_taken_sec"].median() if "time_taken_sec" in df else 30)
    return out


# -------------------------------
# MCQ: Train XGBoost / LightGBM
# -------------------------------
@app.post("/mcq/train")
def train_mcq(payload: MCQTrainPayload):
    df = pd.DataFrame([r.dict() for r in payload.rows])
    if df.empty:
        raise HTTPException(400, "No rows provided")

    feats = _extract_mcq_features(df)
    X = feats.drop(columns=["is_correct"])  # predict correctness
    y = feats["is_correct"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = payload.model.lower()
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss")
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8)
    else:
        raise HTTPException(400, f"Unsupported model: {payload.model}")

    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    f1 = float(f1_score(y_te, preds))

    model_id = f"mcq_{model_name}"
    MODEL_STORE[model_id] = {"clf": clf, "feature_columns": list(X.columns)}

    return {"model_id": model_id, "accuracy": acc, "f1": f1}


@app.post("/mcq/grade")
def grade_mcq(payload: MCQGradePayload):
    """Two modes:
    - If model_id provided and available => use ML model to estimate correctness prob per item and aggregate.
    - Else => rule-based compare to key.
    """
    answers = payload.answers
    key = payload.key

    if payload.model and payload.model in MODEL_STORE:
        entry = MODEL_STORE[payload.model]
        clf = entry["clf"]
        cols = entry["feature_columns"]
        # Build df
        rows = []
        for qid, sel in answers.items():
            rows.append({"choice": sel, "correct_choice": key.get(qid, None)})
        df = pd.DataFrame(rows)
        feats = _extract_mcq_features(df).drop(columns=["is_correct"], errors="ignore")
        # Align columns
        for c in cols:
            if c not in feats.columns:
                feats[c] = 0
        feats = feats[cols]
        probs = clf.predict_proba(feats)[:, 1]
        item_scores = {qid: float(p) for qid, p in zip(answers.keys(), probs)}
        total = float(np.sum(probs))
        return {"mode": "ml", "item_scores": item_scores, "total_estimated_score": total}

    # Rule-based fallback
    correct = 0
    item_scores = {}
    for qid, sel in answers.items():
        is_corr = 1.0 if key.get(qid) == sel else 0.0
        item_scores[qid] = is_corr
        correct += is_corr
    return {"mode": "rule", "item_scores": item_scores, "total_score": float(correct)}


# --------------------------------------
# MCQ Grading from Google Form CSV
# --------------------------------------
@app.post("/mcq/grade-google-form")
def grade_mcq_google_form(
    responses: UploadFile = File(...),
    key_json: Optional[str] = Form(None),
    key_csv: Optional[UploadFile] = File(None),
    student_id_col: Optional[str] = Form(None),
    key_source: Optional[str] = Form(None),   # json | csv | in_sheet
    key_identifier: Optional[str] = Form("KEY"),  # used when key_source == in_sheet
):
    # Load responses CSV or XLSX
    try:
        name = responses.filename or ""
        data = responses.file.read()
        if name.lower().endswith(".xlsx") or name.lower().endswith(".xls"):
            df = pd.read_excel(io.BytesIO(data))
        else:
            df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Failed to read responses file: {e}")

    # Load answer key
    key: Dict[str, str] = {}
    if key_json:
        try:
            key = json.loads(key_json)
        except Exception as e:
            raise HTTPException(400, f"Invalid key_json: {e}")
    elif key_csv is not None:
        try:
            kdf = pd.read_csv(io.BytesIO(key_csv.file.read()))
            # Try common column names
            q_col = None
            a_col = None
            for cand in ["question", "Question", "item", "Item"]:
                if cand in kdf.columns:
                    q_col = cand
                    break
            for cand in ["answer", "Answer", "correct", "Correct"]:
                if cand in kdf.columns:
                    a_col = cand
                    break
            if not q_col or not a_col:
                raise ValueError("Key CSV must contain 'question' and 'answer' columns (case-insensitive)")
            for r in kdf.itertuples(index=False):
                row_vals = r._asdict() if hasattr(r, "_asdict") else dict(zip(kdf.columns, r))
                q = str(row_vals[q_col])
                a = str(row_vals[a_col])
                key[q] = a
        except Exception as e:
            raise HTTPException(400, f"Failed to parse key_csv: {e}")
    elif (key_source or "").lower() == "in_sheet":
        # Build key from a special row in responses identified by key_identifier
        # Determine student id column for locating key row
        sid_col_detect = student_id_col
        if sid_col_detect is None:
            for cand in ["Email Address", "email", "Email", "Name", "Student ID", "student_id"]:
                if cand in df.columns:
                    sid_col_detect = cand
                    break
        if sid_col_detect is None:
            sid_col_detect = df.columns[0]
        ident = (key_identifier or "KEY").strip().lower()
        key_row = None
        for r in df.itertuples(index=False):
            row_dict = r._asdict() if hasattr(r, "_asdict") else dict(zip(df.columns, r))
            val = str(row_dict.get(sid_col_detect, "")).strip().lower()
            if val == ident:
                key_row = row_dict
                break
        if key_row is None:
            raise HTTPException(400, f"Could not find key row in sheet where {sid_col_detect} == '{key_identifier}'")
        # Use all non-empty cells as answers for question columns
        for c in df.columns:
            if c == sid_col_detect:
                continue
            v = key_row.get(c, None)
            if pd.notna(v) and str(v).strip() != "":
                key[c] = str(v)
        if not key:
            raise HTTPException(400, "Extracted empty key from sheet. Ensure key row has answers under each question column.")
    else:
        raise HTTPException(400, "Provide either key_json, key_csv, or set key_source=in_sheet with a key_identifier")

    # Determine student id column
    sid_col = student_id_col
    if sid_col is None:
        for cand in ["Email Address", "email", "Email", "Name", "Student ID", "student_id"]:
            if cand in df.columns:
                sid_col = cand
                break
    if sid_col is None:
        sid_col = df.columns[0]

    # Question columns are those present in key
    questions = [q for q in key.keys() if q in df.columns]
    if not questions:
        raise HTTPException(400, "No overlapping questions between responses and key")

    results = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        sid = str(row_dict.get(sid_col, "unknown"))
        # Skip key row if present in-sheet
        if (key_source or "").lower() == "in_sheet" and sid.strip().lower() == (key_identifier or "").strip().lower():
            continue
        per_q = {}
        correct = 0
        for q in questions:
            selected = str(row_dict.get(q, "")).strip()
            correct_ans = str(key[q]).strip()
            # If key looks like single letter A-D and selected starts with same letter (common pattern)
            letter_match = False
            if len(correct_ans) == 1 and correct_ans.upper() in ["A","B","C","D"]:
                letter_match = selected.strip().upper().startswith(correct_ans.upper())
            is_correct = (selected == correct_ans) or letter_match
            per_q[q] = {"selected": selected, "correct": correct_ans, "is_correct": bool(is_correct)}
            if is_correct:
                correct += 1
        results.append({
            "student": sid,
            "marks": int(correct),
            "max_marks": len(questions),
            "percentage": (100.0 * correct / max(1, len(questions))),
            "per_question": per_q
        })

    return {"graded": len(results), "questions": questions, "results": results}


# --------------------------------------
# Subjective Grading with RoBERTa (STS-B)
# --------------------------------------
SUBJ_MODEL_ID = "cross-encoder/stsb-roberta-base"
_subjective_model = None


def _load_subjective_model():
    global _subjective_model
    if _subjective_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        _subjective_model = {
            "tokenizer": AutoTokenizer.from_pretrained(SUBJ_MODEL_ID),
            "model": AutoModelForSequenceClassification.from_pretrained(SUBJ_MODEL_ID)
        }
    return _subjective_model


@app.post("/subjective/grade")
def grade_subjective(req: SubjectiveGradeRequest):
    mdl = _load_subjective_model()
    tok = mdl["tokenizer"]
    model = mdl["model"]

    import torch
    inputs = tok(req.student_answer, req.reference_answer, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        # STS-B is trained to predict similarity (0..5). Map to 0..max_score
        sim = logits.squeeze().item()
    sim_clamped = max(0.0, min(5.0, sim))
    score = (sim_clamped / 5.0) * req.max_score

    return {
        "similarity_raw": sim,
        "score": float(score),
        "max_score": req.max_score,
        "explanation": "Score derived from semantic similarity between student and reference answers using RoBERTa STS-B cross-encoder."
    }


# --------------------------------------
# Subjective Grading from Handwritten PDF (OCR + RoBERTa)
# --------------------------------------
_ocr_reader = None


def _load_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _pdf_pages_to_images(file_bytes: bytes) -> List["Image.Image"]:
    from PIL import Image
    import fitz  # PyMuPDF
    images = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images


def _ocr_images_to_text(images: List["Image.Image"]) -> str:
    import numpy as np
    reader = _load_ocr_reader()
    texts: List[str] = []
    for img in images:
        arr = np.array(img)
        lines = reader.readtext(arr, detail=0, paragraph=True)
        if isinstance(lines, list):
            texts.append("\n".join(lines))
    return "\n\n".join(texts).strip()


def _score_similarity(student_text: str, reference_text: str, max_score: float) -> float:
    mdl = _load_subjective_model()
    tok = mdl["tokenizer"]
    model = mdl["model"]
    import torch
    inputs = tok(student_text, reference_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        sim = logits.squeeze().item()
    sim_clamped = max(0.0, min(5.0, sim))
    return float((sim_clamped / 5.0) * max_score)


@app.post("/subjective/grade-pdf")
def grade_subjective_pdf(
    pdf: UploadFile = File(...),
    references_json: str = Form(...),  # e.g. {"Q1": "ref answer 1", "Q2": "ref answer 2"}
    max_total: float = Form(100.0),    # scale to 100 by default
):
    try:
        references: Dict[str, str] = json.loads(references_json)
        if not isinstance(references, dict) or not references:
            raise ValueError("references_json must be a non-empty JSON object")
    except Exception as e:
        raise HTTPException(400, f"Invalid references_json: {e}")

    file_bytes = pdf.file.read()
    try:
        images = _pdf_pages_to_images(file_bytes)
        text = _ocr_images_to_text(images)
    except Exception as e:
        raise HTTPException(500, f"Failed to OCR PDF: {e}")

    if not text:
        raise HTTPException(400, "No text extracted from PDF. Ensure the PDF contains clear handwriting.")

    # Score each question then rescale to max_total
    per_question_raw: Dict[str, float] = {}
    for qid, ref in references.items():
        per_question_raw[qid] = _score_similarity(text, ref, 1.0)
    raw_sum = sum(per_question_raw.values()) or 1.0
    per_question: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for qid, raw in per_question_raw.items():
        score = float(raw / raw_sum * max_total)
        per_question[qid] = {"score": score}
        total += score

    return {"total_score": total, "max_total": max_total, "per_question": per_question, "extracted_text_preview": text[:1000]}


# ----------------------------------------
# Handwriting Detection (YOLOv8 + OpenCV)
# ----------------------------------------

# Expect a YOLOv8 model trained for handwriting detection (or text region)
# Provide a default to detect generic text regions via pretrained model if available.
YOLO_PATH = "yolov8n.pt"  # TODO: replace with your trained weights for handwriting/text


def _load_yolo():
    from ultralytics import YOLO
    try:
        return YOLO(YOLO_PATH)
    except Exception as e:
        raise HTTPException(500, f"Failed to load YOLO model at {YOLO_PATH}: {e}")


@app.post("/handwriting/detect")
def detect_handwriting(image: UploadFile = File(...), conf: float = Form(0.25)):
    from PIL import Image
    import cv2
    import numpy as np

    yolo = _load_yolo()
    img_bytes = image.file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = yolo.predict(pil_img, conf=conf, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy().tolist()[0]
            confv = float(b.conf.cpu().numpy().tolist()[0])
            cls = int(b.cls.cpu().numpy().tolist()[0])
            boxes.append({"bbox": xyxy, "confidence": confv, "class_id": cls})

    return {"n": len(boxes), "boxes": boxes}


@app.post("/handwriting/legibility")
def handwriting_legibility(image: UploadFile = File(...)):
    """Simple legibility heuristic using OpenCV: edge density + contrast.
    0..1 higher is more legible.
    """
    import cv2
    import numpy as np

    data = np.frombuffer(image.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, "Invalid image")

    # Edge density
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(edges.mean()) / 255.0

    # Local contrast via Laplacian variance
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    # Normalize contrast to 0..1 using soft scaling
    contrast = 1 - np.exp(-lap_var / (lap_var + 50))

    # Combine
    legibility = float(0.6 * contrast + 0.4 * edge_density)

    return {"edge_density": edge_density, "contrast": contrast, "legibility": legibility}


# -------------------------------
# Forecasting: Prophet / LSTM
# -------------------------------
@app.post("/forecast")
def forecast(req: ForecastRequest):
    df = pd.DataFrame(req.series)
    if "date" not in df.columns or "value" not in df.columns:
        raise HTTPException(400, "series must have 'date' and 'value'")
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df = df.sort_values("date")

    if req.method.lower() == "prophet":
        from prophet import Prophet
        m = Prophet()
        train = df.rename(columns={"date": "ds", "value": "y"})
        m.fit(train)
        future = m.make_future_dataframe(periods=req.horizon)
        fcst = m.predict(future)
        out = fcst.tail(req.horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return {
            "method": "prophet",
            "forecast": [
                {"date": r.ds.strftime("%Y-%m-%d"), "yhat": float(r.yhat), "low": float(r.yhat_lower), "high": float(r.yhat_upper)}
                for r in out.itertuples()
            ]
        }

    elif req.method.lower() == "lstm":
        try:
            import tensorflow as tf
        except Exception as e:
            raise HTTPException(500, f"TensorFlow not installed: {e}")

        values = df["value"].astype(float).values
        # Windowing
        win = 14
        X, y = [], []
        for i in range(len(values) - win):
            X.append(values[i:i+win])
            y.append(values[i+win])
        if not X:
            raise HTTPException(400, "Not enough data for LSTM (need > window)")
        X = np.array(X)[..., None]
        y = np.array(y)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(win,1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=30, batch_size=16, verbose=0)

        # Rollout horizon
        hist = values[-win:].tolist()
        preds = []
        for _ in range(req.horizon):
            x = np.array(hist[-win:])[None, :, None]
            p = float(model.predict(x, verbose=0).squeeze())
            preds.append(p)
            hist.append(p)
        dates = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=req.horizon)
        return {
            "method": "lstm",
            "forecast": [
                {"date": d.strftime("%Y-%m-%d"), "yhat": float(v)} for d, v in zip(dates, preds)
            ]
        }

    else:
        raise HTTPException(400, f"Unknown method: {req.method}")


# -------------------------------
# Clustering: KMeans / DBSCAN
# -------------------------------
@app.post("/cluster")
def cluster(req: ClusterRequest):
    if not req.rows:
        raise HTTPException(400, "No rows provided")

    # Build matrix
    student_ids = []
    feats = []
    for r in req.rows:
        sid = r.get("student_id")
        fdict = r.get("features", {})
        if sid is None:
            raise HTTPException(400, "row missing student_id")
        student_ids.append(sid)
        feats.append([fdict[k] for k in sorted(fdict.keys())])

    X = np.array(feats, dtype=float)
    feature_names = sorted(req.rows[0]["features"].keys())

    if req.algo.lower() == "kmeans":
        k = req.k or 3
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X).tolist()
        centers = km.cluster_centers_.tolist()
        return {"algo": "kmeans", "labels": dict(zip(student_ids, labels)), "centers": centers, "feature_order": feature_names}

    elif req.algo.lower() == "dbscan":
        eps = req.eps or 0.5
        ms = req.min_samples or 5
        db = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X).tolist()
        return {"algo": "dbscan", "labels": dict(zip(student_ids, labels)), "feature_order": feature_names}

    else:
        raise HTTPException(400, f"Unknown algo: {req.algo}")


# -------------------------------
# Health Check & Root
# -------------------------------
@app.get("/")
def root():
    return {"message": "LMS AI Add-ons are running!"}


# -------------------------------
# Optional: simple persistence endpoints (save/load models)
# -------------------------------
@app.post("/models/save")
def save_models():
    blob = {}
    for k, v in MODEL_STORE.items():
        try:
            blob[k] = pickle.dumps(v)
        except Exception:
            blob[k] = None
    # Return base64-ish safe json via hex
    out = {k: (b.hex() if b is not None else None) for k, b in blob.items()}
    return out


@app.post("/models/load")
def load_models(payload: Dict[str, Optional[str]]):
    loaded = 0
    for k, hx in payload.items():
        if hx is None:
            continue
        try:
            MODEL_STORE[k] = pickle.loads(bytes.fromhex(hx))
            loaded += 1
        except Exception:
            pass
    return {"loaded": loaded, "keys": list(MODEL_STORE.keys())}
