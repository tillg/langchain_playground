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
from loguru import logger 
import os

EMBEDDING_MODEL = "nomic-embed-text"
VECTORESTORE_NAME = "a12"

log_dir = 'data/logs'
log_file = f'{log_dir}/app.log'
os.makedirs(log_dir, exist_ok=True)
logger.remove()
logger.add(log_file, colorize=True, enqueue=True)

retriever = get_retriever(vectorstore=get_vectorestore(VECTORESTORE_NAME, embedding_model=EMBEDDING_MODEL))

# retrieved_docs = retriever.invoke(
#     "What are the approaches to Task Decomposition?")
# logger.info(f"No of retrieved docs: {len(retrieved_docs)}, first doc: {
#             retrieved_docs[0].page_content}")

llm = ChatOllama(model="mixtral:8x7b-instruct-v0.1-q8_0")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

Question: {question}

Context: {context}

Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

chat_history = []

# question = "What is Task Decomposition?"
# ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

# second_question = "What are common ways of doing it?"
# ai_msg_2 = rag_chain.invoke(
#     {"input": second_question, "chat_history": chat_history})


# answer = ai_msg_2["answer"]
# logger.info(f"Answer: {answer}")
# logger.info("\n" + pprint.pformat(answer))


while True:
    user_input = input("> ")
    ai_msg = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=user_input), ai_msg["answer"]])
    print("\n Assistant :", ai_msg["answer"])

