# backend_setup.ps1

# Function to check if a command exists
function Test-Command($command) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Check for required tools
$requiredCommands = @("python", "pip")
foreach ($cmd in $requiredCommands) {
    if (-not (Test-Command $cmd)) {
        Write-Error "$cmd is not installed or not in PATH. Please install it and try again."
        exit 1
    }
}

# Create project directory
$projectName = "backend"
New-Item -ItemType Directory -Force -Path $projectName
Set-Location $projectName

# Create virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1

# Install dependencies
pip install fastapi uvicorn[standard] sqlalchemy pydantic pytest httpx python-dotenv

# Create requirements.txt
pip freeze > requirements.txt

# Create project structure
$dirs = @(
    "app", "app/api", "app/api/api_v1", "app/api/api_v1/endpoints",
    "app/core", "app/db", "app/models", "app/schemas", "app/services", "tests"
)
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir
}

# Create main.py
@"
from fastapi import FastAPI
from app.api.api_v1.api import api_router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"@ | Out-File -FilePath "main.py" -Encoding utf8

# Create app/core/config.py
@"
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hyperbolic LLM API"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./hyperbolic_llm.db"
    
    class Config:
        env_file = ".env"

settings = Settings()
"@ | Out-File -FilePath "app/core/config.py" -Encoding utf8

# Create app/db/base.py
@"
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"@ | Out-File -FilePath "app/db/base.py" -Encoding utf8

# Create app/models/message.py
@"
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.db.base import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, index=True)
    response = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
"@ | Out-File -FilePath "app/models/message.py" -Encoding utf8

# Create app/schemas/message.py
@"
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
"@ | Out-File -FilePath "app/schemas/message.py" -Encoding utf8

# Create app/services/llm_service.py
@"
class LLMService:
    @staticmethod
    async def generate_response(content: str) -> str:
        # This is a placeholder for the actual LLM integration
        # In a real implementation, you would call your LLM model here
        return f"LLM response to: {content}"
"@ | Out-File -FilePath "app/services/llm_service.py" -Encoding utf8

# Create app/api/api_v1/endpoints/chat.py
@"
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.base import get_db
from app.models.message import Message
from app.schemas.message import MessageCreate, MessageResponse
from app.services.llm_service import LLMService

router = APIRouter()

@router.post("/chat", response_model=MessageResponse)
async def chat(message: MessageCreate, db: Session = Depends(get_db)):
    llm_response = await LLMService.generate_response(message.content)
    db_message = Message(content=message.content, response=llm_response)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message
"@ | Out-File -FilePath "app/api/api_v1/endpoints/chat.py" -Encoding utf8

# Create app/api/api_v1/api.py
@"
from fastapi import APIRouter
from app.api.api_v1.endpoints import chat

api_router = APIRouter()
api_router.include_router(chat.router, tags=["chat"])
"@ | Out-File -FilePath "app/api/api_v1/api.py" -Encoding utf8

# Create tests/test_chat.py
@"
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from main import app
from app.core.config import settings
from app.db.base import get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_create_message():
    response = client.post(
        f"{settings.API_V1_STR}/chat",
        json={"content": "Hello, AI!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["content"] == "Hello, AI!"
    assert "response" in data
    assert "created_at" in data
"@ | Out-File -FilePath "tests/test_chat.py" -Encoding utf8

# Create .env file
@"
DATABASE_URL=sqlite:///./hyperbolic_llm.db
"@ | Out-File -FilePath ".env" -Encoding utf8

# Create .gitignore
@"
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
ENV/

# IDE
.vscode/
.idea/

# Databases
*.db

# Environment variables
.env

# Logs
*.log

# Test coverage
htmlcov/
.coverage
.coverage.*
.cache

# Pytest
.pytest_cache/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# Initialize git repository
git init
git add .
git commit -m "Initial backend setup"

# Final message
Write-Host "Backend setup complete!" -ForegroundColor Green
Write-Host "To start the server, run 'uvicorn main:app --reload' in this directory." -ForegroundColor Yellow
Write-Host "To run tests, use 'pytest'." -ForegroundColor Yellow