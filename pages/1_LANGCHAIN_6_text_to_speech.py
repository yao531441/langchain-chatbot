import os
import sys
import glob
import utils
import streamlit as st
import numpy as np
import soundfile as sf

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
    
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Speech Chatbot')
st.write('Allows users to interact with the LLM')

class Basic:

    def __init__(self):
        self.openai_model = "speecht5"

    def remove_files(n):
        mp3_files = glob.glob("temp/*mp3")
        if len(mp3_files) != 0:
            for f in mp3_files:
                os.remove(f)
    
    def setup_chat(self):
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=False)
        return llm
    
    def text2speech(self, text, voices, llm):
        prompt = PromptTemplate(template="voices: {voices}, text: {text}", input_variables=["voices","text"])
        print(prompt.format(voices=voices, text=text))
        response = llm.predict(text=prompt.format(voices=voices, text=text))
        response = np.fromstring(response.replace('[', '').replace(']', ''), sep=' ').astype(np.int16)
       
        my_file_name = "audio"
        sf.write(f"temp/{my_file_name}.mp3", response, samplerate=16000)
        return my_file_name

    def main(self):
        llm = self.setup_chat()
        voices = ["BDL", "CLB", "KSP", "RMS", "SLT"]
        option = st.selectbox("Select an voice option", voices, index=1)
        if user_query := st.text_input(label="Ask me anything!", placeholder="Ask me anything!"):
            with st.spinner("Thinking..."):
                result = self.text2speech(user_query, option, llm)
                audio_file = open(f"temp/{result}.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3", start_time=0)

if __name__ == "__main__":
    obj = Basic()
    obj.main()
