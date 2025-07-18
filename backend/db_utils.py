from database import users_collection, chats_collection, pdf_results_collection, db
from models import User, Chat, PDFResult, QueryLog

# Insert a new user
async def insert_user(user: User):
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        return False
    await users_collection.insert_one(user.dict())
    return True

# Get a user by email
async def get_user(email: str):
    return await users_collection.find_one({"email": email})

# Save a chat
async def save_chat(chat: Chat):
    await chats_collection.insert_one(chat.dict())

# Get up to 100 chats for a user
async def get_chats(user_id: str):
    cursor = chats_collection.find({"user_id": user_id})
    return await cursor.to_list(length=100)

# Save a PDF processing result
async def save_pdf_result(pdf_result: PDFResult):
    await pdf_results_collection.insert_one(pdf_result.dict())

# Save a query log (for Q&A and general chat)
querylog_collection = db["querylog"]

async def save_query_log(log: QueryLog):
    await querylog_collection.insert_one(log.dict())
