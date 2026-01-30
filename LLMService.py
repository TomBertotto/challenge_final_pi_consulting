import cohere
import os
from dotenv import load_dotenv

class LLMService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = cohere.Client(api_key)
    
    def detect_domain(self, text: str) -> str:
        prompt_domain = """
            Sos un agente que analiza el dominio de términos y condiciones de un producto.\n
            Tu trabajo es extraer qué dominio representa mejor dicho producto (por ejemplo: financiero, médico, software, educación, ecommerce, otros).\n
            Tu respuesta debe ser una única palabra que indique el dominio.\n
            Los términos y condiciones son:\n
            {text[:4000]}
        """
        try:
            response = self.cohere_client.chat(
                model="command-r-plus-08-2024",
                message=prompt_domain,
                temperature=0
            )
            return response.text.strip()
        except Exception as exc:
            print("LLM ERROR: ", type(exc).__name__)
            raise