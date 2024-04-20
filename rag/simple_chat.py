import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
from vectorestore_factory import get_retriever, get_vectorestore
from langchain.globals import set_debug
import os
from langchain_core.callbacks import FileCallbackHandler
from loguru import logger


EMBEDDING_MODEL = "nomic-embed-text"
VECTORESTORE_NAME = "a12"

log_dir = 'data/logs'
log_file = f'{log_dir}/app.log'
os.makedirs(log_dir, exist_ok=True)
logger.remove()
logger.add(log_file, colorize=True, enqueue=True)

handler = FileCallbackHandler(log_file)
config = {
    'callbacks': [handler]
}

retriever = get_retriever(vectorstore=get_vectorestore(VECTORESTORE_NAME, embedding_model=EMBEDDING_MODEL))
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOllama(model="mixtral:8x7b-instruct-v0.1-q8_0")

def format_docs(docs):
    #logger = logging.getLogger(__name__)
    joined_docs = "\n\n".join(doc.page_content for doc in docs)
    logger.info(f"Joined docs: {joined_docs}")
    return joined_docs


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#set_debug(True)

while True:
    user_input = input("> ")
    try: 
        answer = rag_chain.invoke({"question": user_input})
        print( answer)
    except Exception as e:
        logger.error(f"Error: {e}")
    #print("\n Assistant :", ai_msg["answer"])

