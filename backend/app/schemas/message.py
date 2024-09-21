# app/schemas/message.py
from pydantic import BaseModel
from datetime import datetime

class MessageBase(BaseModel):
    content: str

class MessageCreate(MessageBase):
    pass

class MessageResponse(MessageBase):
    id: int
    response: str
    created_at: datetime

    class Config:
        orm_mode = True
