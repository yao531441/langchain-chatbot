import utils
import streamlit as st
from streaming import StreamHandler

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import ChatMessage, HumanMessage
import base64
import requests
from io import BytesIO

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')
st.write('Allows users to interact with the LLM')

class Basic:

    def __init__(self):
        self.openai_model = "fuyu-8b"
    
    @utils.enable_chat_history
    def main(self):
        image_input = st.file_uploader('Upload a PNG image', type='png')
        image_url = st.text_input("Online Image Link")
        base64_image = None
        if image_input is not None or image_url != "":
            if image_input is not None:
                st.image(image_input)
                base64_image = base64.b64encode(image_input.getvalue()).decode("utf-8")
                base64_image = f"data:image/png;base64,{base64_image}"
            else:
                st.image(BytesIO(requests.get(image_url).content))
                base64_image = image_url
        if not base64_image:
            st.error("Please upload image to continue!")
            st.stop()       
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:                
            utils.display_msg(user_query, 'user')
            message = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_query},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        },
                    ]
                )
            ]
            
            with st.chat_message("assistant"):
                container = st.empty()
                stream_handler = StreamHandler(container)
                llm = ChatOpenAI(openai_api_base = "http://10.0.2.19:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512, callbacks=[stream_handler])
                response = llm(message)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                #container.markdown(response.content)

if __name__ == "__main__":
    obj = Basic()
    obj.main()