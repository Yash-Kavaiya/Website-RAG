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

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = rag_chain.invoke(message)
    return result
css = """
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f0f0f0;
}
.container {
    max-width: 1500px;
    margin: 0 auto;
    background-color: white;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    height: 100px;
}
h1 {
    color: #0078d4;
    text-align: center;
    margin-bottom: 10px;
}
.description {
    text-align: center;
    margin-bottom: 20px;
}
.chatbot-container {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}
.footer {
    text-align: center;
    font-size: 0.9em;
    color: #666;
    margin-top: 20px;
}
.footer a {
    color: #0078d4;
    text-decoration: none;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<div class='container'> <h1>Website RAG</h1> <p class='description'>This website takes a list of URLs and creates a Retrieval-Augmented Generation (RAG) system on top of it. The answers provided are more accurate than previous versions.</p>")
    with gr.Column(elem_classes="chatbot-container"):
        chatbot = gr.ChatInterface(
            respond,
            analytics_enabled=True,
            show_progress="full",
            chatbot=gr.Chatbot(height="70vh"),
            textbox=gr.Textbox(placeholder="Enter your question here...", container=False),

        )
    
    gr.HTML("<div class='footer'>Created by Yash Kavaiya | <a href='https://github.com/Yash-Kavaiya'>GitHub</a> | <a href='https://linkedin.com/in/Yash Kavaiya'>LinkedIn</a></div>")
    gr.HTML("</div>")

if __name__ == "__main__":
    demo.launch(share=True)