from typing import Optional, Dict
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    email: EmailStr
    password: str

    class Config:
        orm_mode = True

class Chat(BaseModel):
    user_email: str
    message: str
    response: str

    class Config:
        orm_mode = True

class PDFResult(BaseModel):
    user_id: Optional[str] = None
    filename: str
    result: Dict
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True



class QueryLog(BaseModel):
    user_id: Optional[str]
    query: str
    answer: str
    source: str  # "pdf_qa" or "general_chat"
    created_at: datetime

