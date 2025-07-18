from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from services import summarize, classify, chatbot_pdf, chatbot_general
from services.pdf_utils import extract_text_from_pdf
from models import PDFResult, QueryLog
from db_utils import save_pdf_result, save_query_log
from database import users_collection
from datetime import datetime
import base64
import traceback

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "✅ Backend is running!"}

# PDF processing request schema
class ProcessRequest(BaseModel):
    filename: str
    filedata: str
    summarize_checked: bool = False
    classify_checked: bool = False
    qa_checked: bool = False
    qa_query: str | None = None
    user_id: str | None = None

@app.post("/process/")
async def process_pdf(request: ProcessRequest = Body(...)):
    summarize_flag = request.summarize_checked
    classify_flag = request.classify_checked
    qa_flag = request.qa_checked

    try:
        content = base64.b64decode(request.filedata)
    except Exception as e:
        print(f"Base64 decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 file data.")

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")
    if not (summarize_flag or classify_flag or qa_flag):
        raise HTTPException(status_code=400, detail="No processing option selected.")

    result = {}

    # Summarization
    if summarize_flag:
        try:
            result["masked_summary"] = summarize.generate_masked_summary(text)
        except Exception as e:
            print(f"Summarization error: {e}")
            raise HTTPException(status_code=500, detail="Summarization failed.")

    # Classification
    if classify_flag:
        try:
            result["classification"] = classify.classify_text(text)
        except Exception as e:
            print("Classification error:", e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Classification failed.")

    # Q&A
    if qa_flag:
        if not request.qa_query or not request.qa_query.strip():
            raise HTTPException(status_code=400, detail="QA query must be provided.")
        try:
            answer = chatbot_pdf.answer_pdf_query(request.qa_query.strip(), text)
            result["pdf_qa_answer"] = answer

            # Save query log for Q&A
            try:
                log = QueryLog(
                    user_id=request.user_id,
                    query=request.qa_query.strip(),
                    answer=answer,
                    source="pdf_qa",
                    created_at=datetime.utcnow()
                )
                await save_query_log(log)
            except Exception as e:
                print(f"Query log save error: {e}")

        except Exception as e:
            print(f"Q&A error: {e}")
            raise HTTPException(status_code=500, detail="Q&A failed.")

    # Save PDF processing result
    try:
        pdf_data = PDFResult(
            user_id=request.user_id,
            filename=request.filename,
            result=result,
            created_at=datetime.utcnow()
        )

        await save_pdf_result(pdf_data)
    except Exception as e:
        print(f"DB save error: {e}")
        # Do not block returning to client if DB save fails

    return result

# General Chat request schema
class ChatRequest(BaseModel):
    query: str
    user_id: str | None = None

@app.post("/general-chat/")
async def general_chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        answer = chatbot_general.answer_general_query(query)

        # Save query log for general chat
        try:
            log = QueryLog(
                user_id=request.user_id,
                query=query,
                answer=answer,
                source="general_chat",
                created_at=datetime.utcnow()
            )
            await save_query_log(log)
        except Exception as e:
            print(f"Query log save error: {e}")

        return {"answer": answer}
    except Exception as e:
        print(f"General Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat failed.")

# User registration request schema
class User(BaseModel):
    email: EmailStr
    password: str

@app.post("/register/")
async def register_user(user: User):
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")
    try:
        await users_collection.insert_one(user.dict())
        return {"message": "Registration successful!"}
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Could not register user.")

@app.post("/login/")
async def login_user(user: User):
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    try:
        existing_user = await users_collection.find_one({"email": user.email})
        if not existing_user or existing_user.get("password") != user.password:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        return {"message": "Login successful!"}
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Could not log in user.")
