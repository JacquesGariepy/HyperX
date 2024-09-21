from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api_v1.api import api_router
from app.core.config import settings
from app.db.base import Base, engine
from backend.app.services.llm_service import LLMService

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://localhost:3000"],  # Ajustez selon vos besoins
    allow_origins=["*"],  # Autorise toutes les origines pour les tests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Créer les tables dans la base de données
Base.metadata.create_all(bind=engine)

# Inclure le router API
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI backend!"}

@app.get("/test-chat")
async def test_chat(message: str):
    # Simuler le comportement de l'endpoint POST /chat
    llm_response = await LLMService.generate_response(message)
    return {"response": llm_response}
# À la fin de votre fichier main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)