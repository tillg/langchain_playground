from typing import Any, Dict, List, Optional
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
import pprint
from chromadb.config import Settings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
import os
from tqdm import tqdm


log_dir = 'data/logs'
log_file = f'{log_dir}/app.log'
os.makedirs(log_dir, exist_ok=True)
logger.remove()
logger.add(log_file, colorize=True, enqueue=True)

DEFAULT_NAME = "default_vectore_store"
VECTORE_STORE_DIR = "data/vectorestores"
DEFAULT_EMBEDDING_MODEL = None


class ChromaPimped(Chroma):
    def __len__(self):
        logger.info(f"Getting length of vectorstore...")
        if  self._collection is None:
                return 0
        all_entries = self.get()
        return len(self.get()['documents'])
    

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, str]] = None, **kwargs: Any) -> List:
        logger.info(f"Starting similarity search with query: {
                    query}, k: {k}, filter: {filter}, kwargs: {kwargs}")

        # Call the parent's method
        results = super().similarity_search(query, k, filter, **kwargs)

        logger.info(f"Similarity search completed. Results: {results}")

        return results

    def add_documents(self, documents, **kwargs: Any) -> List[str]:
        doc_ids = []
        with tqdm(total=len(documents), desc="Adding documents") as pbar:
            for doc in documents:
                doc_id = super().add_documents([doc], **kwargs)
                doc_ids.extend(doc_id)
                pbar.update(1)
        return doc_ids
    
    def delete_all_documents(self):
        logger.info(f"Deleting all documents from vectorstore...")
        all_docs = self.get()
        if len(all_docs['ids']) > 0:
            self.delete(all_docs['ids'])
        logger.info(f"Deleting all documents from vectorstore...Done")

def get_vectorestore(name=DEFAULT_NAME, embedding_model=None):
    persist_directory = os.path.join(VECTORE_STORE_DIR, name)
    logger.info(f"Getting vectorstore for {
                name} with embedding model {embedding_model}...")
    ollama_emb = OllamaEmbeddings()
    if not embedding_model is None:
        ollama_emb = OllamaEmbeddings(
            model=embedding_model)  # For ex. llama2:latest
    vectorstore = ChromaPimped(
        DEFAULT_NAME, ollama_emb, persist_directory=persist_directory)
    logger.info(f"Getting vectorstore for {name} with embedding model {embedding_model}...Done. No of docs: {len(vectorstore)}")
    return vectorstore


def get_retriever(*, name=DEFAULT_NAME, vectorstore=None):
    if vectorstore is None:
        vectorstore = get_vectorestore(name)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}, verbose=True)
    return retriever
