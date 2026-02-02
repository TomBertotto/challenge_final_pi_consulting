from fastapi import FastAPI, HTTPException
from TermsHandler import TermsHandler
from TermsDocument import TermsDocument
import uvicorn
from EmbeddingService import EmbeddingService
from LLMService import LLMService
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from RequestClasses import AskRequest, UploadRequest

app = FastAPI(title = "Challenge Final")
handler = TermsHandler()
embedding_service = EmbeddingService()
llm_service = LLMService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


ENTIDAD_UNICA = "entidad_unica"
ENTIDAD_MULTIPLE = "entidad_multiple"
ENTIDAD_NO_ESPECIFICA = "no_especifica"





if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)



@app.get("/")
def mostrar_index():
    return FileResponse("static/index.html")

@app.post("/upload")
def upload_terms(req: UploadRequest):
    source = req.source
    terms = req.terms
    if not source or not terms:
        raise HTTPException(status_code=400, detail="Se necesitan los términos y condiciones y a quién pertenece")
    
    domain = llm_service.detect_domain(terms)
    terms_document = handler.create_terms(source, terms, domain)

    embedding_service.process_document(terms_document)
    return {
        "message": "Documento cargado",
        "new_terms_added": source,
    }

@app.post("/ask")
def ask(req: AskRequest):
    question = req.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="No se realizó una pregunta")

    question_domain = llm_service.detect_domain(question)
    entities = llm_service.detect_entities(question)

    multi_entidad = entities in (ENTIDAD_MULTIPLE, ENTIDAD_NO_ESPECIFICA)

    results = embedding_service.collection.query(
        query_texts=[question],
        where={"domain": question_domain} if question_domain else None,
        n_results = 30 if multi_entidad else 5
    )

    if not results["documents"] or not results["documents"][0]:
        results = embedding_service.collection.query(
            query_texts = [question],
            n_results=30 if multi_entidad else 5
        )


    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "No hay información relevante para contestar la pregunta",
            "sources": []
        }

    seleccionados = []

    if multi_entidad:
        seleccionados = embedding_service.select_distinct_best_chunks(results, max_entities=3)
        chunks = [c["chunk"] for c in seleccionados]
        sources = [c["source"] for c in seleccionados]
    else:
        chunks = results["documents"][0]
        sources = [meta.get("source") for meta in results["metadatas"][0]]
        
        for ch, src in zip(chunks, sources):
            seleccionados.append({
                "source": src,
                "chunk": ch
            })

    
    contexto = "\n\n".join(
        f"==== PRODUCT: {c['source']} ==== \n {c['chunk']}\n"
        for c in seleccionados
    )

    
    answer = llm_service.answer_question(question, contexto)
    return {
        "question": question,
        "domain": question_domain,
        "entities": entities,
        "answer": answer,
        "sources": sources,
        "context:": contexto
    }

