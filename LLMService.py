import cohere

API_KEY = ""


class LLMService:
    def __init__(self):
        self.client = cohere.Client(API_KEY)
    
    def detect_domain(self, text: str) -> str:
        prompt_domain = """
            Sos un agente que analiza el dominio de términos y condiciones de un producto.\n
            Tu trabajo es extraer qué dominio representa mejor dicho producto (por ejemplo: financiero, médico, software, educación, ecommerce, otros).\n
            Tu respuesta debe ser una única palabra que indique el dominio.\n
            Los términos y condiciones son:\n
            {text[:4000]}
        """

        response = self.client.generate(
            model="",
            prompt=prompt_domain,
            temperature=0
        )
        return response.generations[0].text.strip().lower()