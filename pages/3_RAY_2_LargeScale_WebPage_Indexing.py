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

st.set_page_config(page_title="Document Indexing", page_icon="🌐")
st.header('Large Scale Webpages Indexing')
st.write('Build Knowledge Base based on your documents')

class Custombot:
    def __init__(self):
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('ray_large_web_chat')
        
    def get_actual_links(self, links_str):
        import re
        return re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', links_str)

    
    def document_indexing(self, link_list, db_location):
        # Run the script file
        process = subprocess.Popen(['python', 'my_app/langchain-chatbot/recdp_rag_indexing.py', '--folder', f"{';'.join(link_list)}", '--db_location', db_location, '--type', 'url'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        
        folder = "webpage_db"
        db_location = folder if len(list(glob.glob(folder))) == 1 else None
        
        with st.form("my-form", clear_on_submit=True):
            links = st.text_input(f"Provide Webpage links")
            submitted = st.form_submit_button()
        
        if submitted and links:
            link_list = self.get_actual_links(links)
            with st.expander("Input URL list is"):
                st.write(link_list)
            db_location = db_location if db_location is not None else folder
            with st.spinner("Start Indexing..."):
                self.document_indexing(link_list, db_location)
                st.write(f"Index completed, file saved to {db_location}.")
            for i in glob.glob(db_location):
                db_location = i

        if db_location is not None:
            st.write(f"Pre-generated {db_location} exists, you can start query.")
            if "qa_chain_32" not in st.session_state.keys(): # Initialize the chat engine
                st.session_state.qa_chain_32 = self.setup_qa_chain(db_location, embeddings, llm)
            if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
                self.history_messages.append({"role": "user", "content": prompt})
                with st.chat_message('user'):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st_cb = StreamHandler(st.empty())
                        response = st.session_state.qa_chain_32.run(prompt, callbacks=[st_cb])
                        self.history_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Custombot()
    obj.main()
