import streamlit as st
# import requests
from llama_index.llms.groq import Groq
# import huggingface_hub
from llama_index.core import Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import PromptTemplate
from llama_index.embeddings.gemini import GeminiEmbedding
import dotenv
import os
dotenv.load_dotenv()

GROQ_APIKEY = os.getenv('GROQ')
GOOGLE_GEMINI_KEY = os.getenv('GOOGLE_GEMINI_KEY')

st.title("Language-Assistant")

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        model_name = "models/embedding-001"
        Settings.embed_model = GeminiEmbedding(
            model_name=model_name, api_key=GOOGLE_GEMINI_KEY
        )
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        # index.save_to_disk("./data/index.json")
        Settings.llm = Groq(model="mixtral-8x7b-32768", api_key = GROQ_APIKEY)
        return index

index = load_data()

if "template" not in st.session_state:
    template = (
        "You are a chatbot. Your role is to answer questions regarding the japanese language.\n"
        "The users are generally japanese language learners.\n"
        "You will answer primarily in english and use japanese wherever required \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "You are free to take information from the context. please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    st.session_state.template = qa_template

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask any query about the language"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role":"user", "content":prompt})

    query_engine = index.as_query_engine(text_qa_template=st.session_state.template)
    response = query_engine.query(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role":"assistant", "content":response})