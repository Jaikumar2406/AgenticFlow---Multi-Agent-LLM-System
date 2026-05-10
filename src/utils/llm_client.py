from typing import Optional, List, Dict, Any
from groq import AsyncGroq
from config.settings import settings


class LLMClient:
    """Groq LLM client wrapper"""

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        self.model = settings.GROQ_MODEL

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 4000) -> Dict[str, Any]:
        """Generate text using Groq"""
        if not self.client:
            return {
                "success": False,
                "error": "GROQ_API_KEY not configured"
            }

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": self.model,
                "usage": dict(response.usage) if response.usage else {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """Generate streaming text using Groq"""
        if not self.client:
            return

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"


llm_client = LLMClient()