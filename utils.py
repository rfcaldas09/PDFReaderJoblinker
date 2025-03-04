from pathlib import Path

import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import openai
from dotenv import load_dotenv

import fitz  # PyMuPDF
import os

from configuracao import *

load_dotenv()

#os.environ["OPENAI_API_KEY"] = "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"

openai_api_key = os.getenv("OPENAI_API_KEY")

#PASTA_ARQUIVOS = Path(__file__).parent / 'arquivos'
PASTA_ARQUIVOS = Path('arquivos')
os.makedirs(PASTA_ARQUIVOS, exist_ok=True)  # Cria o diretório se não existir

def extract_text_from_pdf(pdf_path):
    # Usando PyMuPDF (fitz) para extrair texto
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text("text")
    return text


def importacao_documentos():
    documentos = []
    for arquivo in PASTA_ARQUIVOS.glob('*.pdf'):
        texto = extract_text_from_pdf(str(arquivo))
        
        # Criando documentos com o formato correto
        doc = Document(page_content=texto, metadata={'source': str(arquivo)})
        documentos.append(doc)
    
    return documentos


def split_de_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500,
        separators=["/n\n", "\n", ".", " ", ""]
    )
    
    # Já não é mais necessário transformar os documentos, pois são objetos do tipo Document
    documentos = recur_splitter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata['source'] = doc.metadata['source'].split('/')[-1]
        doc.metadata['doc_id'] = i
    return documentos


def cria_vector_store(documentos):
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    print(f"API Key: {openai_api_key}")

    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store


def cria_chain_conversa():
    documentos = importacao_documentos()
    documentos = split_de_documentos(documentos)
    vector_store = cria_vector_store(documentos)

    chat = ChatOpenAI(model=get_config('model_name'))
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )
    retriever = vector_store.as_retriever(
        search_type=get_config('retrieval_search_type'),
        search_kwargs=get_config('retrieval_kwargs')
    )
    prompt = PromptTemplate.from_template(get_config('prompt'))
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    st.session_state['chain'] = chat_chain
