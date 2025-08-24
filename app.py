import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os

# Load your Google API key from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

# helper function to read text from uploaded PDFs
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Helper function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    st.set_page_config(page_title="PDF Chatbot Task")
    st.header("RAG Chatbot for PDF Documents")

    # sidebar for uploading files
    with st.sidebar:
        st.title("Your Documents")
        pdf_files = st.file_uploader("Upload your 2 PDF files and click 'Submit'", accept_multiple_files=True)
        
        if st.button("Submit"):
            if pdf_files:
                with st.spinner("Processing your documents..."):
                    # Step 1: Get all the text from the PDFs
                    raw_text = get_pdf_text(pdf_files)
                    
                    # Step 2: Split text into smaller chunks
                    text_chunks = get_text_chunks(raw_text)
                    # check to see if we got anything
                    
                    # Step 3: Create embeddings and the vector store
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                    
                    # Save the processed data in the session so we dont have to reprocess
                    st.session_state.vector_store = vector_store
                    st.success("Done! You can now ask questions.")
            else:
                st.warning("You need to upload at least one PDF.")

    # Main area for conversation
    st.write("Ask a question about the content of your documents:")
    
    # Check if the vector store is ready before allowing questions
    if "vector_store" in st.session_state and st.session_state.vector_store:
        user_question = st.text_input("Your question:")
        if user_question:
            vector_store = st.session_state.vector_store
            
            # Find relevant documents based on the question
            docs = vector_store.similarity_search(user_question)
            
            # Define the prompt and the conversational chain
            prompt_template = """
            Answer the question as detailed as possible from the provided context.
            If the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.

            Context:\n {context}\n
            Question: \n{question}\n

            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Get the response and display it
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("### Answer:")
            st.write(response["output_text"])
    else:
        st.info("Upload PDFs and click 'Submit' to get started.")

if __name__ == "__main__":
    main()