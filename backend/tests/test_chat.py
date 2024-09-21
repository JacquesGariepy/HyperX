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
