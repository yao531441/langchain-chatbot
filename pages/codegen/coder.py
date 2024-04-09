from functools import lru_cache

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from pages.codegen.config import emb_model_path, index_path, example_scala_code
from pages.codegen.prompt_config import rag_suffix, convert_to_cpp_temp, example_temp

import re


@lru_cache()
def get_embedding(model_path):
    embedding = HuggingFaceEmbeddings(
        model_name=model_path,
    )
    return embedding


@lru_cache(maxsize=10)
def retrieve_reference(code_query):
    embeddings = get_embedding(emb_model_path)
    db = FAISS.load_local(index_path, embeddings)
    retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'lambda_mult': 1})
    matched_documents = retriever.get_relevant_documents(query=code_query)
    results = ''
    for document in matched_documents:
        results += document.page_content + "\n"
    return results


def extract_code(text):
    print(text)
    pattern = r'```c\+\+|```cpp|\```C\+\+|```'
    parts = re.split(pattern, text)
    return parts[1]


def generate_to_cpp_code(llm, code_text):
    convert_to_cpp_prompt = convert_to_cpp_temp.format(code_text)
    res = llm.invoke(convert_to_cpp_prompt)
    cpp_code = extract_code(res.content)

    return cpp_code


def generate_velox_udf(llm, cpp_code, rag_text=''):
    prompt = example_temp + cpp_code + "\n```"
    if bool(rag_text):
        # reference = retrieve_reference(cpp_code)
        prompt = prompt + rag_suffix.format(rag_text)
    res = llm.invoke(prompt)
    return res.content


def generate_prompt(llm, code_text, rag_flag=False):
    convert_to_cpp_prompt = generate_to_cpp_prompt(code_text)
    res = llm.invoke(convert_to_cpp_prompt)
    cpp_code = extract_code(res.content)
    print(cpp_code)
    prompt = example_temp + cpp_code + "\n```"
    if rag_flag:
        reference = retrieve_reference(cpp_code)
        prompt = prompt + rag_suffix.format(reference)
    return prompt


