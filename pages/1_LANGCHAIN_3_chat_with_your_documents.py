import os
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

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your documents')
st.write('Has access to custom documents and can respond to user queries by referring to the content within those documents')

class CustomDataChatbot:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('rag_chat')

    def save_file(self, folder, file):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path
    
    def load_data(self, to_find):
        docs = []
        for file in glob.glob(to_find):
            loader = PyPDFLoader(file)
            docs.extend(loader.load())         
        return docs

    def setup_qa_chain(self, to_find, embeddings, llm):
        with st.spinner(text="Loading documents."):
            # Load documents
            docs = self.load_data(to_find)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )        
        
        # indexing
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            splits = text_splitter.split_documents(docs)
            vectordb = Chroma.from_documents(splits, embeddings)

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

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        embeddings = self.load_embedding()
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", temperature=0.2, streaming=True, max_tokens=512)
        
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
                st.error("Please upload PDF documents to continue!")
                st.stop()

            if uploaded_files:
                st.write(f"System Log: Received files.")
                file_path = []
                for file in uploaded_files:
                    file_path.append(self.save_file(folder, file))
                st.write(f"System Log: Saved files.")
 
        if "qa_chain" not in st.session_state.keys(): # Initialize the chat engine
            st.session_state.qa_chain = self.setup_qa_chain(to_find, embeddings, llm)

        st.write(f"Ready for Question!")
        if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
            self.history_messages.append({"role": "user", "content": prompt})
            with st.chat_message('user'):
                st.write(prompt)

        # If last message is not from assistant, generate a new response
        if self.history_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_cb = StreamHandler(st.empty())
                    response = st.session_state.qa_chain.run(prompt, callbacks=[st_cb])
                    self.history_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
