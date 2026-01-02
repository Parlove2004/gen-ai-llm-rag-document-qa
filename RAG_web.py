import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os

keyo=os.getenv('openai_key')
keyg=os.getenv('gemini_key')


llm=ChatOpenAI(model='gpt-4',api_key=keyo)

emb_llm=GoogleGenerativeAIEmbeddings(model="text-embedding-004",
                                     google_api_key=keyg)


st.title("RAG Pipeline")
file=st.file_uploader("Select PDF:",type=['pdf','txt'])

if file is not None:
    # Save to current working directory
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
        st.success(f'{file.name} uploaded successfully')
    
    if file.name.endswith(".pdf"):
        loder=PyMuPDFLoader(file_path=file.name)
        docs=loder.load()
    else:
        loder=TextLoader(file_path=file.name)
        docs=loder.load()

    splitter=RecursiveCharacterTextSplitter(
    chunk_size=file.size*.15,
    chunk_overlap=file.size*.015)

    chunks=splitter.split_documents(docs)

    vs=FAISS.from_documents(embedding=emb_llm,documents=chunks)

    query=st.text_input("Question:")

    retriever = vs.as_retriever(search_kwargs={'k':5}) 
    rdocs=retriever.invoke(query)

    context_text='\n\n'.join([doc.page_content for doc in rdocs])

    prompt=PromptTemplate(
    template='''You are a helpful assistant.
    Answer only from the provided context.
    If context is insufficiant,just say information not found
    
    context = {context}
    question = {question}
    ''',input_variables=['context','question'])

    chain=prompt|llm
    res=chain.invoke({'context':context_text,'question':query})
    st.write(res.content)