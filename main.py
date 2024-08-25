import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'websiteRAG'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
import gradio as gr
def respond(message, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("./Vector_DB/faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}| prompt| llm | StrOutputParser())
    result=rag_chain.invoke(message)
    return result
demo = gr.ChatInterface(respond)

if __name__ == "__main__":
    demo.launch()
