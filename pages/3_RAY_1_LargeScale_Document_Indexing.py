import os, sys
import utils
import streamlit as st
from streaming import StreamHandler

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import subprocess

st.set_page_config(page_title="Document Indexing", page_icon="📄")
st.header('Large Scale Document Indexing')
st.write('Build Knowledge Base based on your documents')

class Custombot:
    def __init__(self):
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('ray_large_chat')

    def save_file(self, folder, file):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path
    
    def document_indexing(self, files_path, db_location):
        # Run the script file
        process = subprocess.Popen(['python', 'my_app/langchain-chatbot/recdp_rag_indexing.py', '--folder', files_path, '--db_location', db_location], stdout=subprocess.PIPE, stderr=sys.stdout.buffer)
        with st.expander("Open to see progress"):
            #stdout, stderr = result.communicate()
            with process.stdout:
                for line in iter(process.stdout.readline, b''):
                    st.text(line.decode("utf-8").strip())

    def load_embedding(self):
        # Create embeddings and store in vectordb
        if ('RECDP_CACHE_HOME' not in os.environ) or (not os.environ['RECDP_CACHE_HOME']):
            os.environ['RECDP_CACHE_HOME'] = os.path.join(os.getcwd(), "models")
        #embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_model_name = "all-MiniLM-L6-v2"
        local_embedding_model_path = os.path.join(os.environ['RECDP_CACHE_HOME'], embedding_model_name)
        print(local_embedding_model_path)
        if os.path.exists(local_embedding_model_path):
            embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
        else:
            raise FileNotFoundError(local_embedding_model_path)
        return embeddings
 
    def setup_qa_chain(self, db_location, embeddings, llm):
        vectordb = Chroma(persist_directory=db_location,
                            embedding_function=embeddings)
        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        embeddings = self.load_embedding()
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", temperature=0.2, streaming=True, max_tokens=512)
        
        folder = st.text_input("Server File Location:", placeholder="./large_scale_data/")
        if not folder:
            st.error("Please confirm the location of files")
            st.stop()

        to_find = os.path.join(folder, '*.pdf')
        to_find_db = f"{folder}_db"
        num_files = len(list(glob.glob(to_find)))
        db_location = None
        for i in glob.glob(to_find_db):
            db_location = i

        with st.expander("Upload more documents"):
            # User Inputs
            uploaded_files = st.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
            if uploaded_files:
                st.write(f"System Log: Received new files.")
                file_path = []
                for file in uploaded_files:
                    file_path.append(self.save_file(folder, file))
                st.write(f"System Log: Saved files.")
            num_files = len(list(glob.glob(to_find)))

        if num_files > 0:
            col1, col2 = st.columns((5, 1))
            col1.write(f"{num_files} files uploaded for Document Indexing, click 'Start Index' to create index.")
            if col2.button("Start Index"):
                db_location = db_location if db_location is not None else f"{folder}_db"
                with st.spinner("Start Indexing..."):
                    self.document_indexing(folder, db_location)
                    st.write(f"Index completed, file saved to {db_location}.")
                for i in glob.glob(db_location):
                    db_location = i

        if db_location is not None:
            st.write(f"Pre-generated {db_location} exists, you can start query.")
            if "qa_chain_31" not in st.session_state.keys(): # Initialize the chat engine
                st.session_state.qa_chain_31 = self.setup_qa_chain(db_location, embeddings, llm)
            if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
                self.history_messages.append({"role": "user", "content": prompt})
                with st.chat_message('user'):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st_cb = StreamHandler(st.empty())
                        response = st.session_state.qa_chain_31.run(prompt, callbacks=[st_cb])
                        self.history_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Custombot()
    obj.main()
