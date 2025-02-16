from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import os
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize FastAPI app
app = FastAPI()
port = int(os.environ.get("PORT", 5000))

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
vector_store = None
conversation_chain = None

# Input data schema
class Query(BaseModel):
    query: str
    session_id: str

def get_pdf_text(file):
    pdf_reader = PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    prompt_template = """
    You are an AI assistant tasked with answering questions based on the content of a PDF document. 
    Your goal is to provide detailed, accurate, and relevant information from the document.

    Given the following context and question, please follow these steps:
    1. Carefully analyze the context and identify key information relevant to the question.
    2. If the context contains relevant information, use it to formulate a comprehensive answer.
    3. If the context doesn't contain directly relevant information, clearly state that the information is not available in the document and suggest asking a question related to the document's content.
    4. Do not use your general knowledge to answer questions not related to the document.
    5. If you're unsure or the information is ambiguous, acknowledge this and suggest rephrasing the question or asking about a topic covered in the document.
    6. Always maintain a professional and helpful tone.

    Context:
    {context}

    Human: {question}
    AI Assistant: Let me analyze the information from the document and provide you with an answer.
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return conversation_chain

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store, conversation_chain
    if file.filename.split('.')[-1].lower() != 'pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    content = get_pdf_text(file)
    text_chunks = get_text_chunks(content)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)
    return {"message": "File processed successfully"}

@app.post("/query")
async def query(query: Query):
    global conversation_chain
    if not conversation_chain:
        raise HTTPException(status_code=400, detail="Please upload a PDF file first")
    
    try:
        response = conversation_chain.invoke({"question": query.query})
        return {"answer": response["answer"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
