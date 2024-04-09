import time

import utils
import streamlit as st

from pages.codegen.coder import generate_prompt, generate_to_cpp_code, generate_velox_udf, retrieve_reference
from pages.codegen.config import demo_sample_python_code
from streaming import StreamHandler

# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

st.set_page_config(page_title="Gluten_Coder_Chatbot_V2", page_icon="💬")
st.header('Gluten Coder Chatbot')
st.write('Convert code to Gluten/Velox UDF with the LLM')
from code_editor import code_editor

code_editor_btns_config = [{
    "name": "Copy",
    "feather": "Copy",
    "hasText": True,
    "alwaysOn": True,
    "commands": ["copyAll",
                 ["infoMessage",
                  {
                      "text": "Copied to clipboard!",
                      "timeout": 2500,
                      "classToggle": "show"
                  }
                  ]
                 ],
    "style": {"top": "0rem", "right": "0.4rem"}
}, {
    "name": "Run",
    "feather": "Play",
    "primary": True,
    "hasText": True,
    "showWithIcon": True,
    "commands": ["submit"],
    "style": {"bottom": "0.44rem", "right": "0.4rem"}
}
]

info_bar = {
    "name": "input code",
    "css": "\nbackground-color: #bee1e5;\n\nbody > #root .ace-streamlit-dark~& {\n   background-color: #444455;\n}\n\n.ace-streamlit-dark~& span {\n   color: #fff;\n    opacity: 0.6;\n}\n\nspan {\n   color: #000;\n    opacity: 0.5;\n}\n\n.code_editor-info.message {\n    width: inherit;\n    margin-right: 75px;\n    order: 2;\n    text-align: center;\n    opacity: 0;\n    transition: opacity 0.7s ease-out;\n}\n\n.code_editor-info.message.show {\n    opacity: 0.6;\n}\n\n.ace-streamlit-dark~& .code_editor-info.message.show {\n    opacity: 0.5;\n}\n",
    "style": {
        "order": "1",
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "center",
        "width": "100%",
        "height": "2.5rem",
        "padding": "0rem 0.6rem",
        "padding-bottom": "0.2rem",
        "margin-bottom": "-1px",
        "borderRadius": "8px 8px 0px 0px",
        "zIndex": "9993"
    },
    "info": [
        {
            "name": "Your code",
            "style": {
                "width": "800px"
            }
        }
    ]
}


class Basic:

    def __init__(self):
        self.openai_model = "deepseek-coder:33b-instruct"
        self.init = True
        self.llm = ChatOpenAI(openai_api_base="http://localhost:11434/v1", model_name=self.openai_model,
                              openai_api_key="not_needed",
                              # max_tokens=2048,
                              streaming=True)


    def main(self):
        # col_d, col_e, col_f = st.columns([1, 1, 1])
        # mode_list = ['c_cpp', 'java', 'python', 'scala']
        # language = col_d.selectbox("lang:", mode_list, index=mode_list.index("scala"))
        # theme = col_e.selectbox("theme:", ["default", "light", "dark", "contrast"])
        # shortcuts = col_f.selectbox("shortcuts:", ["emacs", "vim", "vscode", "sublime"], index=2)

        response_dict = code_editor('', height=(8, 20), lang='scala', theme='dark',
                                    shortcuts='vscode', focus=False, buttons=code_editor_btns_config,
                                    info=info_bar,
                                    props={"style": {"borderRadius": "0px 0px 8px 8px"}},
                                    options={"wrap": True})
        code_to_convert = response_dict['text']

        if bool(code_to_convert):
            print(code_to_convert)

            phase_1_reminder = "Converting your code to C++..."
            with st.chat_message("ai"):
                st.write(phase_1_reminder)
            with st.spinner(phase_1_reminder):
                cpp_code = generate_to_cpp_code(self.llm, code_to_convert)
                # time.sleep(10)
                with st.chat_message("ai"):
                    st.write(cpp_code)

            phase_2_reminder = "Retrieve reference from velox document and code..."
            with st.chat_message("ai"):
                st.write(phase_2_reminder)
            with st.spinner(phase_2_reminder):
                rag_source = retrieve_reference(cpp_code)
                with st.chat_message("ai"):
                    st.write(rag_source)

            phase_3_reminder = "Converting the C++ code to velox based udf..."
            with st.chat_message("ai"):
                st.write(phase_3_reminder)
            with st.spinner(phase_3_reminder):
                result = generate_velox_udf(self.llm, cpp_code)
                with st.chat_message("ai"):
                    st.write(result)


if __name__ == "__main__":
    obj = Basic()
    obj.main()
