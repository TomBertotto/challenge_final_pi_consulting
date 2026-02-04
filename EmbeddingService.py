import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import cohere
from TermsDocument import TermsDocument



class EmbeddingService:
    def __init__(self, collection_name: str = "terms_and_conditions"):
        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = cohere.Client(api_key)

        cohere_ef = embedding_functions.CohereEmbeddingFunction(
            api_key = api_key,
            model_name="embed-multilingual-v3.0"
        )
        
        self.db_client = chromadb.PersistentClient(
            path="chroma_db"
        )

        self.collection = self.db_client.get_or_create_collection(name=collection_name, embedding_function=cohere_ef)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 50
        )
    
    def process_document(self, terms_document: TermsDocument):
        text = terms_document.get_terms()
        raw_chunks = self.text_splitter.split_text(text)
        chunks = self._merge_small_chunks(raw_chunks)
        ids = []
        documents = []
        metadatas = []    

        for i, chunk in enumerate(chunks):
            ids.append(f"{terms_document.terms_id}_{i}")
            documents.append(chunk)
            metadatas.append({
                "terms_id" : terms_document.terms_id,
                "source": terms_document.source,
                "domain" : terms_document.domain,
                "chunk_index": i
            })
        
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    
    def select_distinct_best_chunks(self, results, max_entities = 15):
        best_by_terms_id = {}

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            tid = meta["terms_id"]

            if tid not in best_by_terms_id or dist < best_by_terms_id[tid]["distance"]:
                best_by_terms_id[tid] = {
                    "chunk": doc,
                    "terms_id": tid,
                    "distance": dist,
                    "source": meta.get("source"),
                    "domain": meta.get("domain")
                }
                
                

        seleccionados = sorted(
            best_by_terms_id.values(),
            key = lambda x: x["distance"]
        )[:max_entities]

        return seleccionados

    def _merge_small_chunks(self, chunks, min_length = 200):
        merged = []
        buffer = ""
        for chunk in chunks:
            if len(buffer) < min_length:
                buffer += " " + chunk
            else:
                merged.append(buffer.strip())
                buffer = chunk
        
        if buffer:
            merged.append(buffer.strip())
        return merged