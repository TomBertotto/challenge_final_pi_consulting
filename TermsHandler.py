import os
import uuid
from pathlib import Path
from TermsDocument import TermsDocument

PATH = "archivos_locales/"

class TermsHandler:
    def __init__(self):
        self.base_path = PATH
        os.makedirs(self.base_path, exist_ok=True)

    def create_terms(self, source: str, terms, domain: str) -> TermsDocument:
        terms_id = str(uuid.uuid4())
        file_path = f"{PATH}{source}_{terms_id}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(terms)

        return TermsDocument(source=source, terms=terms, domain=domain, terms_id= terms_id)

    def get_terms(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read
