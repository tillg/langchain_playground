import bs4
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
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
from tqdm import tqdm
import os
from vectorestore_factory import get_vectorestore

log_dir = 'data/logs'
log_file = f'{log_dir}/app.log'
os.makedirs(log_dir, exist_ok=True)
logger.remove()
logger.add(log_file, colorize=True, enqueue=True)


DIRECTORY_TO_INGEST = "data/ingest/a12_small"
EMBEDDING_MODEL = "nomic-embed-text"
VECTORESTORE_NAME = "a12"


loader = DirectoryLoader(
    DIRECTORY_TO_INGEST, glob="**/*.md", show_progress=True)

logger.info(f"Loading documents...")
docs = loader.load()
logger.info(f"Loading documents...Done {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

logger.info(f"No of splits: {len(all_splits)}")

vectorstore = get_vectorestore(
    VECTORESTORE_NAME, embedding_model=EMBEDDING_MODEL)
logger.info(f"Length of vectorstore before all: {
            len(vectorstore)}")
vectorstore.delete_all_documents()
logger.info(f"Length of vectorstore after deleting all documents: {
            len(vectorstore)}")
logger.info(f"Adding docs to vectorstore...")
vectorstore.add_documents(all_splits)
logger.info(f"Adding docs to vectorstore...Done {len(vectorstore)}")

logger.info(f"Persisting vectorstore...")
vectorstore.persist()
logger.info(f"Persisting vectorstore...Done {len(vectorstore)}")
