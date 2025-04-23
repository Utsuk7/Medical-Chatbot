import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template,
    )
    return custom_prompt

def load_llm(huggingface_repo_id,HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.1,
        model_kwargs={
            "max_length": 512,
            "token": HF_TOKEN
        },
        task="text-generation"
    )
    return llm


def main():
    st.title("Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    

    prompt=st.chat_input("Pass Your query here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        

        custom_prompt_template = """use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, say "I don't know", don't try to make up an answer.
        Dont provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please."""


        HF_TOKEN = os.environ.get("HF_TOKEN")
        huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"
        

        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Vectorstore not found. Please check the path.")
            
            qa_chain= RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=huggingface_repo_id,HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
                return_source_documents=True,
            )

            response=qa_chain.invoke({"query": prompt})

            result=response["result"]
            source_docs=response["source_documents"]
            result_to_show=result+"\n\n\n\n"+"Source Document"+str(source_docs)

        #response = "Hii !, Im your AI Healthcare assistant. How may I help you ?"
            st.chat_message('assistant').markdown(result_to_show) 
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})


        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
    