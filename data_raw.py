from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import csv

# Configure logging
logging.basicConfig(filename='rag_chain.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'websiteRAG'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def read_urls_from_csv(file_path):
    urls = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            urls.append(row[0])
    return urls

urls = read_urls_from_csv('./Data/urls.csv')
print("URLs loaded:", urls)
logging.info(f"URLs loaded: {urls}")

loader = WebBaseLoader(urls)
docs = loader.load()
logging.info(f"Number of documents loaded: {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
logging.info(f"Number of splits created: {len(splits)}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local("./Vector_DB/faiss_index")
logging.info("Vector store created and saved")

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = ChatGoogleGenerativeAI(model="gemini-pro")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
question = "Give me All Topics of Create a question answering solution by using Azure AI Language"
print("Question:", question)
logging.info(f"Question: {question}")

answer = rag_chain.invoke(question)
print("Answer:", answer)
logging.info(f"Answer: {answer}")