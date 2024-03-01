import utils
import streamlit as st
from streaming import StreamHandler

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import pytesseract
import os
from PIL import Image
from io import BytesIO
import base64
    
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Form to Table')
st.write('Allows users to interact with the LLM')

class Basic:

    def __init__(self):
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('form_convert')

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        with st.form("my-form", clear_on_submit=True):
            image_input = st.file_uploader('Upload a jpg image')
            submitted = st.form_submit_button()

        if submitted and image_input is not None:
            base64_image = base64.b64encode(image_input.getvalue()).decode("utf-8")
            base64_image = f"data:image/png;base64,{base64_image}"
            image_md = f"![image]({base64_image})"
            self.history_messages.append({"role": "user", "content": image_md})
            with st.chat_message('user'):
                st.write(image_md)
            text = pytesseract.image_to_string(Image.open(BytesIO(image_input.getvalue())))
            
            user_query = f"Please convert 'Input' into a markdown table.\n Input: {text}"
            message = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_query},
                    ]
                )
            ]
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    stream_handler = StreamHandler(st.empty())
                    llm = ChatOpenAI(openai_api_base = "http://10.0.2.14:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512, callbacks=[stream_handler])
                    response = llm(message)
                    self.history_messages.append({"role": "assistant", "content": response.content})
                    image_input = None

if __name__ == "__main__":
    obj = Basic()
    obj.main()
