from fastapi import FastAPI, HTTPException
from TermsHandler import TermsHandler
from TermsDocument import TermsDocument
import uvicorn
from EmbeddingService import EmbeddingService
from LLMService import LLMService
import os
from datetime import datetime

app = FastAPI(title = "Challenge Final")
handler = TermsHandler()
embedding_service = EmbeddingService()
llm_service = LLMService()

@app.post("/upload")
def upload_terms(source, terms: str):
    if not source or not terms:
        raise HTTPException(status_code=400, detail="Se necesitan los términos y condiciones y a quién pertenece")
    
    domain = llm_service.detect_domain(terms)
    terms_document = handler.create_terms(source, terms, domain)

    embedding_service.process_document(terms_document)

@app.post("/ask")
def ask(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="No se realizó una pregunta")

    question_domain = llm_service.detect_domain(question)

    results = embedding_service.collection.query(
        query_texts=[question],
        where={"domain": question_domain},
        n_results = 5
    )

    if not results["documents"] or not results["documents"][0]:
        results = embedding_service.collection.query(
            query_texts = [question],
            n_results=5
        )

    chunks = results["documents"][0]

    if not chunks:
        return {
            "answer" : "No hay información relevante para contestar la pregunta",
            "fuentes" : []
        }

    answer = llm_service.answer_question(question, chunks)
    return {
        "question": question,
        "domain": question_domain,
        "answer": answer,
        "sources": chunks
    }

