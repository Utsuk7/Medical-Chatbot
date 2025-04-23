import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS


# Step 1: Setup LLM (Mistral with HUggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    temperature=0.1,
    model_kwargs={
        "max_length":512,
        "token":HF_TOKEN
        },
    task="text-generation"
    )
    return llm


#Step 2: Connect LLM with FAISS and Create chain
DB_FAISS_PATH="vectorstore/db_faiss"
custom_prompt_template = """use the pieces of information provided in the context to answer user's question.
If you don't know the answer, say "I don't know", don't try to make up an answer.
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please."""

def set_custom_prompt():
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template,
    )
    return custom_prompt


# model used for creating embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#LOAD Database
db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": set_custom_prompt()},
    return_source_documents=True,
)

#Invoke with single query
user_query = input("Write your query: ")
response = qa_chain.invoke({"query": user_query})
print(f"Answer: {response['result']}")
print(f"Source Documents: {response['source_documents']}")