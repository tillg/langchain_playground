# langchain_playground

Playing around with [langchain](https://www.langchain.com).


# Dev Setup & Running

I use a couple of windows with a standard command (this all assumes you are in the project folder):

```bash
# Follow the application log 
less +F data/logs/app.log

# Follow the 
less +F ~/.ollama/logs/server.log
```

To run a script I use the module syntax: `python -m rag.simple_chat`.

# History

## 2024-04-20 Configurable vectorestores
* Vectorestores are now in config file
* Config of a configuration contains also the model and the directory with the data to ingest
* Use a 

## 2024-04-20 Debugging LangChain chains
* Direct logging into a file so it doesn't interfere with my dialog in the terminal.

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

* Which model to use for local embeddings with Ollama? [In this blog entry (April 8, 2024)](https://ollama.com/blog/embedding-models) they give an overview.
* For having iTerm layouts with commands so I get back my screen setup: [iTomate](https://github.com/kamranahmedse/itomate)
* For a later version: [Building a Confluence Q&A App with LangChain and ChatGPT](https://www.shakudo.io/blog/building-confluence-kb-qanda-app-langchain-chatgpt). Hint: There is a [ConfluenceLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.confluence.ConfluenceLoader.html) in LangChain!!
* While looking for a fast embedding model: [Local Embedding done right - Medium](https://medium.com/@alekseyrubtsov/local-embedding-done-right-5a8bf129ec42)
