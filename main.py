import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os

def main():
    load_dotenv()

    st.header("Chat PDF")

    pdf = st.file_uploader(type="pdf", label="Upload a PDF file", accept_multiple_files=False, key=None)

    if pdf:
        pdf_reader = PdfReader(pdf)

        # extract text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,  chunk_overlap=200, length_function=len)

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        # create Embeddings
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as file:
                VectorStore = pickle.load(file)
            st.write("Embeddings read from disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as file: 
                pickle.dump(VectorStore, file)
            st.write("embeddings created and saved to disk")

        # accept question query 
        query = st.text_input("Enter your question here", key="question")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb: 
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            
            st.write(response)


        

if __name__ == "__main__":
    main()