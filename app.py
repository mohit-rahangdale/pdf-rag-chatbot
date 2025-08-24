import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os

# Load your Google API key 
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


genai.configure(api_key=GOOGLE_API_KEY)


def get_pdf_text(pdf_files):
    """Reads and extracts all text from a list of PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits a large text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#gradio Functions
def process_documents(files):
    """The main processing pipeline. Triggered by the 'Process' button."""
    if not files or len(files) != 2:
        return None, gr.update(value="Please upload two PDF files.", visible=True)
    
    try:
        # The whole RAG process in one go
        raw_text = get_pdf_text(files)
        text_chunks = get_text_chunks(raw_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Success message shown in status box
        return vector_store, gr.update(value="Documents processed successfully! You can now ask questions.", visible=True)
    except Exception as e:
        print(e)  # For debugging in the logs
        return None, gr.update(value=f"An error occurred: {e}", visible=True)


def user_interaction(user_input, chat_history, vector_store):
    """Handles the chat logic for each user message."""
    if vector_store is None:
        # Show warning inside chat history itself
        chat_history.append((user_input, "Please process your documents first."))
        return "", chat_history
    
    # Find relevant documents and run the QA chain
    docs = vector_store.similarity_search(user_input)
    
    
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
    
    response = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
    
    # Add the user's message and the bot's response to the history
    chat_history.append((user_input, response["output_text"]))
    return "", chat_history

#gradio Interface definition
with gr.Blocks(title="PDF RAG Chatbot") as demo:
    vector_store_state = gr.State()

    gr.Markdown("# RAG Chatbot for Two PDF Documents ")
    gr.Markdown("Upload two PDF files, click 'Process', and then ask questions about their content.")

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.Files(
                label="Upload your 2 PDF files", 
                file_count="multiple", 
                file_types=[".pdf"]
            )
            process_button = gr.Button("Process Documents", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, visible=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation")
            user_textbox = gr.Textbox(label="Ask a question...", interactive=True)
            
    # When the process button is clicked, run the processing function
    process_button.click(
        fn=process_documents,
        inputs=[file_uploader],
        outputs=[vector_store_state, status_box]
    )
    
    # When the user submits a message, run the chat function
    user_textbox.submit(
        fn=user_interaction,
        inputs=[user_textbox, chatbot, vector_store_state],
        outputs=[user_textbox, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
