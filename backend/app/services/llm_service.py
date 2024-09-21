# app/services/llm_service.py
import aiohttp
from app.core.config import settings

class LLMService:
    @staticmethod
    async def generate_response(content: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.LLM_API_URL, json={"prompt": content}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "No response from LLM")
                else:
                    return f"Error: Unable to get response from LLM (Status: {response.status})"