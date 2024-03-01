import asyncio

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index import SimpleDirectoryReader
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_openai import OpenAI
import glob
import utils

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")

model_name = "mistral-7b-instruct-v0.2"
endpoint = "http://localhost:8000/v1"

class CustomDataChatbot:
    def __init__(self):
        self.history_messages = utils.enable_chat_history('llama_chat')

    def save_file(self, folder, file):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path
    
    def load_embedding(self):
        if ('RECDP_CACHE_HOME' not in os.environ) or (not os.environ['RECDP_CACHE_HOME']):
            os.environ['RECDP_CACHE_HOME'] = os.path.join(os.getcwd(), "models")
        #embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_model_name = "all-MiniLM-L6-v2"

        # Create embeddings and store in vectordb
        local_embedding_model_path = os.path.join(os.environ['RECDP_CACHE_HOME'], embedding_model_name)
        print(local_embedding_model_path)
        if os.path.exists(local_embedding_model_path):
            embeddings = HuggingFaceEmbedding(model_name=local_embedding_model_path)
        else:
            raise FileNotFoundError(local_embedding_model_path)
        return embeddings
    
    def load_data(self, folder):
        reader = SimpleDirectoryReader(input_dir=folder, recursive=True)
        docs = reader.load_data()           
        return docs

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        llm = OpenAI(
            openai_api_base=endpoint,
            model_name=model_name,
            openai_api_key="not_needed",
            streaming=True,
        )
        
        embedding = self.load_embedding()
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding)
        
        folder = st.text_input("Server File Location:", placeholder="./data/")
        if not folder:
            st.error("Please confirm the location of files")
            st.stop()
        to_find = os.path.join(folder, '*.pdf')
        
        is_uploaded = len(list(glob.glob(to_find)))>0
        if not is_uploaded:
            # User Inputs
            uploaded_files = st.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
            if not uploaded_files:
                st.error("Please upload PDF to build knowledge base")
                st.stop()

            if uploaded_files:
                st.write(f"System Log: Received files.")
                file_path = []
                for file in uploaded_files:
                    file_path.append(self.save_file(folder, file))
                st.write(f"System Log: Saved files.")
        docs = self.load_data(folder=folder)
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        st.write(f"Ready for Question!")
        if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

        if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
            self.history_messages.append({"role": "user", "content": prompt})
            with st.chat_message('user'):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.stream_chat(prompt)
                    response_str = ""
                    response_container = st.empty()
                    for token in response.response_gen:
                        response_str += token
                        response_container.write(response_str)
                    message = {"role": "assistant", "content": response.response}
                    self.history_messages.append(message) # Add response to message history
                
if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
