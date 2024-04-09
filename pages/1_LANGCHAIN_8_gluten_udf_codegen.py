import utils
import streamlit as st

from pages.codegen.coder import generate_prompt
from streaming import StreamHandler

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

st.set_page_config(page_title="Gluten_Coder_Chatbot", page_icon="💬")
st.header('Gluten Coder Chatbot')
st.write('Convert code to Gluten/Velox UDF with the LLM')


class Basic:

    def __init__(self):
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('basic_chat')

    def setup_chain(self):
        # llm = ChatOpenAI(openai_api_base="http://localhost:8000/v1", model_name=self.openai_model,
        llm = ChatOpenAI(openai_api_base="http://localhost:11434/v1", model_name="deepseek-coder:33b-instruct",
                         openai_api_key="not_needed",
                         # max_tokens=2048,
                         streaming=True)
        chain = ConversationChain(llm=llm, verbose=True)
        self.llm = llm

        return chain

    def main(self):
        chain = self.setup_chain()
        for message in self.history_messages:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_query := st.chat_input(placeholder="Ask me anything!"):
            # self.history_messages.append({"role": "user", "content": user_query})

            with st.chat_message('user'):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_cb = StreamHandler(st.empty())
                    new_prompt = generate_prompt(self.llm, user_query)

                    response = chain.run(new_prompt, callbacks=[st_cb])
                    # self.history_messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    obj = Basic()
    obj.main()