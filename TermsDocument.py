class TermsDocument:
    def __init__(self, source, terms, domain, terms_id: str):
        self.source = source
        self.terms = terms
        self.terms_id = terms_id
        self.domain = domain


    def get_source(self) -> str:
        return self.source

    def get_terms(self) -> str:
        return self.terms
    
    def get_metadata(self) -> dict:
        return {
            "terms_id" : self.terms_id,
            "domain" : self.domain,
            "source" : self.source,
        }