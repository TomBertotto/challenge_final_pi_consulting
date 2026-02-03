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
            *)Contexto: \n
                {chunks}\n
            *)Respuesta:\n
            """
    
    def answer_question(self, question, chunks: str) -> str:
        system_prompt="""
            *)Sos un agente que debe responder preguntas sobre documentación de términos y condiciones.
            *)La respuesta debe realizarse en base a segmentos de información (contexto) y respetando las siguientes reglas al responder:\n
                - No usar emojis\n
                - Contestar siempre en español independientemente de si la pregunta y el contexto están en otro idioma\n
                - Si la pregunta es sensible u ofensiva responder únicamente: "No puedo contestar esa pregunta"\n
                - La respuesta debe realizarse en base al contexto y segmentos de información proporcionados
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

    def detect_entities(self, question) -> str:
        prompt_entities = """
            *)Sos un agente que analiza preguntas y determina si una pregunta determinada está solicitando una respuesta que involucre una sola entidad o muchas.\n\n
            - Si la pregunta pide resultados de una sola cosa entonces se trata de una entidad unica
            - Si la pregunta pide diferentes ejemplos o resultados sobre un tópico en particular entonces se trata de entidades múltiples
            - Si no especifica ninguna entonces no se determina
            *)El formato de tu respuesta debe ser una palabra a elección de las siguientes que mejor se ajuste a la cantidad de entidades (no incluir puntuación):\n
            - entidad_unica
            - entidad_multiple
            - no_especifica
        """
        prompt_question = f"""
            La pregunta es:\n
            {question}
        """
        try:
            response = self.cohere_client.chat(
                model="command-r-plus-08-2024",
                messages=[
                    {"role": "system", "content" : prompt_entities},
                    {"role": "user", "content": prompt_question}
                    ],
                temperature=0
            )
            return response.message.content[0].text.lower()
        except Exception as exc:
            print("LLM ERROR: ", type(exc).__name__)
            raise

