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
from rag.vectorestore_factory import get_retriever, get_vectorestore
from langchain.globals import set_debug
import os
from langchain_core.callbacks import FileCallbackHandler
from loguru import logger
from rag.handlerToDocumentLog import HandlerToDocumentLog

VECTORESTORE_NAME = "a12_large_mxbai"  # "a12_small"

log_dir = 'data/logs'
log_file = f'{log_dir}/app.log'
os.makedirs(log_dir, exist_ok=True)
logger.remove()
logger.add(log_file, colorize=True, enqueue=True)

#handler = FileCallbackHandler(log_file)
handler = HandlerToDocumentLog()
config = {
    'callbacks': [handler]
}

retriever = get_retriever(VECTORESTORE_NAME)
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOllama(model="mixtral:8x7b-instruct-v0.1-q8_0")

def format_docs(docs):
    #logger = logging.getLogger(__name__)
    joined_docs = "\n\n".join(doc.page_content for doc in docs)
    logger.info(f"Joined {len(docs)} docs to one doc of length {len(joined_docs)}")
    return joined_docs

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def chat_loop():
    while True:
        user_input = input("> ")
        try: 
            answer = rag_chain.invoke({"question": user_input},  {
                                    "callbacks": [handler]})
            print( answer)
        except Exception as e:
            logger.error(f"Error: {e}")
        #print("\n Assistant :", ai_msg["answer"])


if __name__ == '__main__':
    chat_loop()
