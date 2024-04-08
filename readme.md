# langchain_playground

Playing around with [langchain](https://www.langchain.com).

## 2024-04-06 Doing RAG tutorial

* Folllowing [this tutorial from langchain](https://python.langchain.com/docs/use_cases/question_answering/quickstart/)

## 2024-04-08 Ingesting 

* Trying to ingest the A12 documentation, ~ 3'800 markdown docs.
* Trying to ingest with llama2:latest. Takes long:
  *  3456/19666 [30:02<2:23:58,  1.88it/s]
  *  Expected time > 3 hours
*  With default embedding model ()
   *  1224/19666 [10:28<2:46:50,  1.84it/s]
   *  Expected time > 3 hours
*  With embedding model nomic-embed-text
   *  15 minutes! 🥰

## Tech reading

* While looking for a fast embedding model: [Local Embedding done right - Medium](https://medium.com/@alekseyrubtsov/local-embedding-done-right-5a8bf129ec42)