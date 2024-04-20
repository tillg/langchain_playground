from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from vectorestore_factory import get_vectorestore
from langchain_community.embeddings import OllamaEmbeddings
import logging

EMBEDDING_MODEL = "nomic-embed-text"
VECTORESTORE_NAME = "a12"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Initializing llm...")
llm = ChatOllama(model="mixtral:8x7b-instruct-v0.1-q8_0")
logger.info(f"Initializing llm...Done")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
logger.info(f"Loading documents...")
docs = loader.load()
logger.info(f"Loading documents...Done {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OllamaEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

# cleanup
vectorstore.delete_collection()
