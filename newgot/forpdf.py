# from gpt4all import GPT4All
# import os


# model = GPT4All(model_name="mistral-7b-openorca.Q4_0.gguf",
#                 model_path= os.getcwd() +"\model")

# with model.chat_session():
#   print("What is your question?")
#   output = model.generate(input())
#   print(output)
#   #print(model.current_chat_session)


## All the important imports

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ChromaDB for data vector storage
import chromadb
# Embedded to get transformer model
from chromadb.utils import embedding_functions



# local path for model
local_path = ("models/gpt4all-falcon-newbpe-q4_0.gguf")


# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Get the book and load it into the loader, and we will split the text into chunks
# loader = TextLoader('./docs/book.pdf')
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
# texts = text_splitter.split_documents(documents)
# print(texts)
# where the file is located
filepath = ("./docs/revenue.pdf")


# use Chromadb to get a path to save the data to and store
chroma_client = chromadb.PersistentClient(path="my_vectordb")

loader = PyPDFLoader(filepath)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
texts = text_splitter.split_documents(documents)
#print(texts)

# This is the model for the embeddings we can change this around depending on the type of embedding model we require
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")


# We have to split the metadata and the actual text in the collective text extracted
actual_texts = [doc.page_content for doc in texts]
metadatas = [doc.metadata for doc in texts]

# we can create a collection here for chromadb, this can be using this function which can get or create dependin on if the
# Collection exists already
collection = chroma_client.get_or_create_collection(name="new_collection", embedding_function=sentence_transformer_ef)


## Print debugs
# print(len(texts))
# print(len(metadatas))
# print(len(actual_texts))

## In order to add to the collection we need id and this can be created using the length of the texts
id_array = []
for i in range(len(texts)):
  id_array.append("id"+str(i+1))

# I have commented the next lines as we don't need to add but we can add the text, the metadata and the ids to the collection
# vector database
collection.add(
               documents=actual_texts,
               metadatas=metadatas,
               ids = id_array)

# this is the straight method to query anything we use. This will require the query text and the n_results we want to take back.
# Finally we can give some includes in the return , some exampls are documents, metadata, ids. and so on
# results = collection.query(
#     query_texts="Romeo and Juliet",
#     n_results=10,
#     include=['documents']
# )

# print(results['documents'])


# # Verbose is required to pass to the callback manager
#TODO: Find out the reason for the callbacks, it might not be needed.
#llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)

llm = GPT4All(model=local_path)

# Here we add the question we want to ask, currently it is hard coded but this can be
# set up with sessions so we can have context aware
question = "What were the Percent of Status Determinations of newly established accounts made within 90 days from the last day of the quarter in which the account first became liable?"


# we can collect the results using the query searching
results = collection.query(
  query_texts=question,
  n_results=2,
  include=['documents', 'metadatas']
)


# we can use this standard template to help the ai answer this question and also give it in the correct format back to us
template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: You can list it or you put it into a paragraph, keep it to 400 words maximum."""

# we can get the context using that query
context = "\n".join(str(x) for x in results['documents'])
# for the prompt we can provide all the inputs
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
# invoke the chain and ask the question
llm_chain = LLMChain(prompt=prompt, llm=llm)
answers = llm_chain.invoke(question)
# get the answers
print(answers['text'])

