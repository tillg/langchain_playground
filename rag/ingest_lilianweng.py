import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
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

from vectorestore_factory import get_vectorestore

EMBEDDING_MODEL = "nomic-embed-text"
VECTORESTORE_NAME = "lilianweng"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

logger.info(f"Doc loaded[:500]: {docs[0].page_content[:500]}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

logger.info(f"No of chunks: {len(all_splits)}")
logger.info(f"Chunk #1: {len(all_splits[0].page_content)}")
logger.info(f"Metada of my chunks: {all_splits[10].metadata}")

vectorstore = get_vectorestore(VECTORESTORE_NAME, embedding_model=EMBEDDING_MODEL)
logger.info(f"Length of vectorstore before all: {
            len(vectorstore)}")
vectorstore.delete_all_documents()
logger.info(f"Length of vectorstore after deleting all documents: {
            len(vectorstore)}")
vectorstore.add_documents(all_splits)
vectorstore.persist()
logger.info(f"Length of vectorstore after adding docs: {
            len(vectorstore)}")
