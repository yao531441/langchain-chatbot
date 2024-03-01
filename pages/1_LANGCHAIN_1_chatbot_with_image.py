import utils
import streamlit as st
from streaming import StreamHandler

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import ChatMessage, HumanMessage
import base64

    
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Basic Chatbot')
st.write('Allows users to interact with the LLM')

class Basic:

    def __init__(self):
        self.openai_model = "fuyu-8b"
        self.history_messages = utils.enable_chat_history('image_chat')

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        image_input = st.file_uploader('Upload a PNG image')
        image_url = st.text_input("Online Image Link")
        base64_image = None
        if image_input is not None or image_url != "":
            if image_input is not None:
                base64_image = base64.b64encode(image_input.getvalue()).decode("utf-8")
                base64_image = f"data:image/png;base64,{base64_image}"
            else:
                base64_image = image_url
            image_md = f"![image]({base64_image})"
        if not base64_image:
            st.error("Please upload image to continue!")
            st.stop()

        self.history_messages.append({"role": "user", "content": image_md})
        with st.chat_message('user'):
            st.write(image_md)
                
        if user_query := st.chat_input(placeholder="Ask me anything!"):
            self.history_messages.append({"role": "user", "content": user_query})
            with st.chat_message('user'):
                st.write(user_query)
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
                with st.spinner("Thinking..."):
                    stream_handler = StreamHandler(st.empty())
                    llm = ChatOpenAI(openai_api_base = "http://10.0.2.19:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512, callbacks=[stream_handler])
                    response = llm(message)
                    self.history_messages.append({"role": "assistant", "content": response.content})
            base64_image = None

if __name__ == "__main__":
    obj = Basic()
    obj.main()
