from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import random
import json
from pathlib import Path
import openai
app = FastAPI(title="LMS AI Assignment Service")
router = APIRouter()
DATA_DIR = Path("data")
ASSIGNMENTS_DIR = DATA_DIR / "assignments"
ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
# ---------- Utility: Load CSVs ----------
def _load_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path.name}: {e}")

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    for c in df.columns:
        lc = c.lower()
        for name in candidates:
            if name.lower() in lc:
                return c
    return None

def load_data():
    activity = _load_csv_safely(DATA_DIR / "student_activity.csv")
    scores = _load_csv_safely(DATA_DIR / "student_scores.csv")
    content = _load_csv_safely(DATA_DIR / "student_content.csv")

    topic_col = find_col(content, ["topic", "module", "chapter", "unit"])
    if topic_col:
        content[topic_col] = content[topic_col].astype(str).str.strip().str.lower()

    return activity, scores, content

try:
    activity_df, scores_df, content_df = load_data()
except Exception as e:
    activity_df, scores_df, content_df = None, None, None
    print("Startup warning:", e)

# ---------- Weak-topic analysis ----------
def require_columns(df: pd.DataFrame, required_any_of: Dict[str, List[str]]) -> Dict[str, str]:
    mapping = {}
    for logical, options in required_any_of.items():
        col = find_col(df, options)
        if not col:
            raise ValueError(
                f"Could not find a column for '{logical}'. Looked for: {options}. Available: {list(df.columns)}"
            )
        mapping[logical] = col
    return mapping

def compute_weak_topics(scores: pd.DataFrame, student_id: str, top_k: int = 3) -> List[str]:
    mapping = require_columns(scores, {
        "student_id": ["student_id", "sid", "student", "user_id"],
        "topic": ["topic", "module", "chapter", "unit"],
        "score": ["score"],
        "max_score": ["max_score", "total", "full_score"]
    })

    sdf = scores.copy()
    sdf[mapping["student_id"]] = sdf[mapping["student_id"]].astype(str).str.strip()
    sdf[mapping["topic"]] = sdf[mapping["topic"]].astype(str).str.strip().str.lower()
    sdf = sdf[sdf[mapping["student_id"]] == str(student_id).strip()]

    if sdf.empty:
        return []

    sdf["accuracy"] = sdf[mapping["score"]] / sdf[mapping["max_score"]]
    weak_topics = (
        sdf.groupby(mapping["topic"])["accuracy"]
        .mean()
        .sort_values(ascending=True)
        .head(top_k)
        .index.tolist()
    )
    return weak_topics


# ---------- MCQ Distractor Generator ----------
def generate_distractors(correct_answer: str, topic_rows: pd.DataFrame, term_col: str, def_col: str, count: int = 3):
    """
    Generate plausible wrong answers for MCQs by using other definitions from the same topic.
    If not enough are available, generate simple meaningful variations.
    """
    # Take definitions of other terms from the same topic
    other_defs = topic_rows[topic_rows[def_col] != correct_answer][def_col].dropna().tolist()
    random.shuffle(other_defs)

    # Take up to 'count' distractors from other definitions
    distractors = other_defs[:count]

    # If not enough, create variations of the correct answer (split by comma or words)
    while len(distractors) < count:
        words = correct_answer.split()
        # Pick 2-3 keywords and slightly modify
        fake_answer = " ".join(random.sample(words, min(len(words), 2)))
        if fake_answer not in distractors and fake_answer != correct_answer:
            distractors.append(fake_answer)

    return distractors[:count]



# ---------- MCQ generator ----------
def generate_mcqs_for_topic(content: pd.DataFrame, topic: str, target_count: int = 5) -> List[Dict]:
    topic_col = find_col(content, ["topic", "module", "chapter", "unit"])
    term_col = find_col(content, ["term", "keyword"])
    def_col = find_col(content, ["definition", "meaning"])
    diff_col = find_col(content, ["difficulty", "level"])

    if not topic_col or not term_col or not def_col:
        return []

    rows = content[content[topic_col].str.strip().str.lower() == topic.strip().lower()]
    mcqs = []

    if rows.empty:
        # Fallback placeholder question with shuffled correct index
        answer_index = random.randint(0, 3)
        options = ["Option A", "Option B", "Option C", "Option D"]
        mcqs.append({
            "topic": topic.capitalize(),
            "question": f"Placeholder question for {topic.capitalize()}",
            "options": options,
            "answer_index": answer_index,
            "difficulty": "Medium"
        })
        return mcqs

    for _, row in rows.iterrows():
        if len(mcqs) >= target_count:
            break

        term = str(row[term_col])
        correct_answer = str(row[def_col])
        difficulty = row[diff_col] if diff_col and pd.notna(row[diff_col]) else "Medium"

        try:
            # OpenAI prompt
            prompt = f"""
            You are a teacher. Create one multiple-choice question (MCQ) for the topic "{topic}".
            Term: {term}
            Definition: {correct_answer}
            Difficulty: {difficulty}

            The question should:
            - Ask about the meaning or role of the term.
            - Provide 4 options (1 correct, 3 distractors).
            - Mark the correct answer with its index.

            Return strictly JSON with keys: "question", "options", "answer_index", "difficulty".
            """
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            ai_output = response.choices[0].message["content"]
            mcq = json.loads(ai_output)

            # Ensure correct answer is in options and shuffle
            options = mcq.get("options", [])
            if correct_answer not in options:
                distractors = options[:3] if len(options) >= 3 else ["Incorrect definition 1", "Incorrect definition 2", "Incorrect definition 3"]
                answer_index = random.randint(0, 3)
                options = distractors.copy()
                options.insert(answer_index, correct_answer)
            else:
                answer_index = options.index(correct_answer)

            mcq["options"] = options[:4]
            mcq["answer_index"] = answer_index
            mcq["topic"] = topic.capitalize()
            mcqs.append(mcq)

        except Exception:
            # Fallback using meaningful distractors
            distractors = generate_distractors(correct_answer, rows, term_col, def_col)
            answer_index = random.randint(0, 3)
            options = distractors.copy()
            options.insert(answer_index, correct_answer)
            options = options[:4]

            mcqs.append({
                "topic": topic.capitalize(),
                "question": f"What is '{term}' in {topic.capitalize()}?",
                "options": options,
                "answer_index": answer_index,
                "difficulty": difficulty
            })


    return mcqs

# ---------- API Schemas ----------
class AssignmentRequest(BaseModel):
    student_id: str
    num_questions: int = 10
    per_topic: int = 5
    weak_topics: Optional[int] = 3

class Assignment(BaseModel):
    student_id: str
    topics: List[str]
    questions: List[Dict]

class BulkRequest(BaseModel):
    num_questions: int = 10
    per_topic: int = 5
    weak_topics: Optional[int] = 3

# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reload-data")
async def reload_data():
    global activity_df, scores_df, content_df
    activity_df, scores_df, content_df = load_data()
    return {
        "status": "reloaded",
        "activity_rows": len(activity_df),
        "scores_rows": len(scores_df),
        "content_rows": len(content_df)
    }

@router.post("/generate", response_model=Assignment)
async def generate_assignment(payload: AssignmentRequest):
    global activity_df, scores_df, content_df
    if activity_df is None or scores_df is None or content_df is None:
        try:
            activity_df, scores_df, content_df = load_data()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    student_id = str(payload.student_id)
    weak_k = payload.weak_topics or 3

    try:
        # Compute weak topics
        topics = compute_weak_topics(scores_df, student_id, top_k=weak_k)

        # Fallback: all topics from content_df if weak topics empty
        if not topics:
            topic_col = find_col(content_df, ["topic", "module", "chapter", "unit"])
            if topic_col:
                topics = content_df[topic_col].dropna().astype(str).str.strip().str.lower().unique().tolist()
            else:
                topics = ["General"]

        per_topic = max(1, int(payload.per_topic))
        questions: List[Dict] = []

        for t in topics:
            questions.extend(generate_mcqs_for_topic(content_df, t, target_count=per_topic))

        num_q = max(1, int(payload.num_questions))
        if len(questions) > num_q:
            random.shuffle(questions)
            questions = questions[:num_q]

        out_path = ASSIGNMENTS_DIR / f"{student_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "student_id": student_id,
                "topics": topics,
                "questions": questions
            }, f, indent=2)

        return Assignment(student_id=student_id, topics=topics, questions=questions)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-all")
async def generate_all(payload: BulkRequest):
    global activity_df, scores_df, content_df
    if activity_df is None or scores_df is None or content_df is None:
        try:
            activity_df, scores_df, content_df = load_data()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    sid_col = find_col(scores_df, ["student_id", "sid", "student", "user_id"])
    if not sid_col:
        raise HTTPException(status_code=400, detail="Could not find a student_id column in student_scores.csv")

    student_ids = sorted(set(scores_df[sid_col].astype(str).tolist()))
    results = {}
    for sid in student_ids:
        req = AssignmentRequest(
            student_id=sid,
            num_questions=payload.num_questions,
            per_topic=payload.per_topic,
            weak_topics=payload.weak_topics
        )
        assignment = await generate_assignment(req)  # type: ignore
        results[sid] = assignment

    return {"generated_for": len(results), "students": list(results.keys())}

# Mount router
app.include_router(router, prefix="/assignments", tags=["assignments"])
