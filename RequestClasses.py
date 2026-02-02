from pydantic import BaseModel

class AskRequest(BaseModel): #TODO: refactor
    question: str

class UploadRequest(BaseModel):
    source: str
    terms: str
