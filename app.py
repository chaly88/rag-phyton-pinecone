import streamlit as st
import os
from utils import *
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Importamos el paquete recien instalado
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY'] = 'AIzaSyAP_Shl0W8eJ9wgfeNd42mfu9HdcZwt6Yw'
genai.configure(api_key = GOOGLE_API_KEY)


FILE_LIST = "archivos.txt"

st.set_page_config(page_title='RAG PINECONE')
st.header("RAG PINECONE", divider='rainbow')
# Lista con los ficheros cargados
# archivos = []
OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')
st.subheader('Pregunta', divider='rainbow')

with st.sidebar:
    archivos = load_name_files(FILE_LIST)
    files_uploaded = st.file_uploader(
        "Carga tu archivo",
        type="pdf",
        accept_multiple_files=True
        )
    
    if st.button('Procesar Documentos'):
        for pdf in files_uploaded:
            if pdf is not None and pdf.name not in archivos:
                archivos.append(pdf.name)
                text_to_pinecone(pdf)
        archivos = save_name_files(FILE_LIST,archivos)

    if len(archivos)>0:
        st.write('Archivos cargados')
        lista_documentos = st.empty()
        with lista_documentos.container():
            for arch in archivos:
                st.write(arch)
            if st.button('Borrar documentos'):
                archivos = []
                clean_files(FILE_LIST)
                lista_documentos.empty()

if len(archivos)>0:
    userQuestion = st.text_input("Realiza tu pregunta")
    if userQuestion:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        vstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

        docs = vstore.similarity_search(userQuestion, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=userQuestion)

        st.write(respuesta)
        # Costo
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=userQuestion)
            st.write(cb)

st.header("GEMINI", divider='rainbow')
if len(archivos)>0:
    userQuestionGemini = st.text_input("Realiza tu pregunta",key='input_gemini')
    if userQuestionGemini:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        vstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

        docs = vstore.similarity_search(userQuestionGemini, 3)

        # Define una variable model, usando gemini-pro

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=userQuestionGemini)

        st.write(respuesta)
