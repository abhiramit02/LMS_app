from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import docx
import pdfplumber
from groq import Groq

router = APIRouter()
client = Groq(api_key="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")


def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text

def extract_text_from_docx(path: str) -> str:
    import docx
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

@router.post("/generate-from-file")
async def generate_from_file(file: UploadFile = File(...), num_questions: int = 5):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        os.remove(file_path)
        return {"error": "Only PDF and DOCX files are supported"}

    os.remove(file_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    input_text = text[:3000]
    prompt = f"""
    Generate {num_questions} multiple-choice questions from the text below, each with:
    - A clear question
    - Four options (A, B, C, D)
    - Correct answer letter

    Text:
    {input_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return {"file": file.filename, "mcqs": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")
