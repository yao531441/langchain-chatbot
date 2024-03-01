import utils
import streamlit as st

from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
    
st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Chatbot with Internet Access')
st.write('Equipped with internet access, enables users to ask questions about recent events')

class ChatbotTools:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('web_chat')

    def setup_agent(self):
        # Define tool
        ddg_search = DuckDuckGoSearchRun(max_results=1)
        tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions",
            )
        ]

        # Setup LLM and Agent
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512,)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            #verbose=True
        )
        return agent

    def main(self):
        agent = self.setup_agent()
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if user_query := st.chat_input(placeholder="Ask me anything!"):
            self.history_messages.append({"role": "user", "content": user_query})
            with st.chat_message('user'):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_cb = StreamlitCallbackHandler(st.container())
                    response = agent.run(user_query, callbacks=[st_cb])
                    self.history_messages.append({"role": "assistant", "content": response})
                    st.write(response)

if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
