import cohere
import os
from dotenv import load_dotenv

class LLMService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = cohere.ClientV2(api_key)
    
    def detect_domain(self, text: str) -> str:
        prompt_domain = """
            Sos un agente que analiza el dominio de términos y condiciones de un producto.\n
            Tu trabajo es extraer qué dominio representa mejor dicho producto (por ejemplo: financiero, médico, software, educación, ecommerce, otros).\n
            Tu respuesta debe ser una única palabra que indique el dominio.\n
        """
        prompt_terms = f"""
            Los términos y condiciones son:\n
            {text[:4000]}
        """
        try:
            response = self.cohere_client.chat(
                model="command-r-plus-08-2024",
                messages=[
                    {"role": "system", "content" : prompt_domain},
                    {"role": "user", "content": prompt_terms}
                    ],
                temperature=0
            )
            return response.message.content[0].text.lower()
        except Exception as exc:
            print("LLM ERROR: ", type(exc).__name__)
            raise
    

    def _generate_user_prompt(self, question, chunks) -> str:
        return f"""
            *)Pregunta: \n
                {question} \n
            *)Contexto-Segmentos de información: \n
                {chunks}\n
            *) Respuesta:\n
            """
    
    def answer_question(self, question, chunks: str) -> str:
        system_prompt="""
            *)Sos un agente que debe responder una pregunta en base a diferentes segmentos de información y respetando las siguiente reglas en tu respuesta:\n
                - No usar emojis\n
                - Contestar siempre en español independientemente de si la pregunta y el contexto están en otro idioma\n
                - Si la pregunta es sensible u ofensiva responder: "No puedo contestar esa pregunta"\n
                - La respuesta debe siempre centrada usando los segmentos de información provistos
        """

        respuesta = self.cohere_client.chat(
            model="command-r-plus-08-2024",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._generate_user_prompt(question, chunks)}
            ]
        )
        return respuesta.message.content[0].text
    

