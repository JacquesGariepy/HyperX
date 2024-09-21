# app/api/api_v1/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.base import get_db
from app.models.message import Message
from app.schemas.message import MessageCreate, MessageResponse
from app.services.llm_service import LLMService

router = APIRouter()

@router.post("/chat", response_model=MessageResponse)
async def chat(message: MessageCreate, db: Session = Depends(get_db)):
    try:
        llm_response = await LLMService.generate_response(message.content)
        db_message = Message(content=message.content, response=llm_response)
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        return db_message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))