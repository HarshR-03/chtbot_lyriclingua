import streamlit as st
import dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from query import load_vector_store, combine_vector_stores
from api_tool import search_jp_vocabulary


dotenv.load_dotenv()

GOOGLE_GEMINI_KEY = os.getenv('GOOGLE_API_KEY')

st.title("Language-Assistant")

@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner(text="Loading Gemini 2.5 Flash model – hang tight!"):

        llm = init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            api_key=GOOGLE_GEMINI_KEY
        )
        llm.bind_tools(search_jp_vocabulary)
        return llm

@st.cache_resource(show_spinner=False)
def load_combined_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store1 = load_vector_store("grammar1", embedding_model)
    store2 = load_vector_store("grammar2", embedding_model)
    combined_store = combine_vector_stores([store1, store2])
    faiss_indices = set(range(combined_store.index.ntotal))
    mapping_indices = set(combined_store.index_to_docstore_id.keys())
    missing = faiss_indices - mapping_indices
    if missing:
        print(f"Missing indices in index_to_docstore_id: {missing}")
    return combined_store

llm = load_model()
vector_store= load_combined_vector_store()


if "template" not in st.session_state:
    template = ( 
        "You are a japanese language teacher. Your role is to answer the user's query.\n"
        "The users are generally japanese music listeners.\n"
        "You can take use of the tool for looking up japanese vocabulary. It takes input a single word in japanese script"
        "You will answer primarily in english and use japanese wherever required.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "You are free to take information from the context. please answer the question: {query_str}\n"
    )
    qa_template = ChatPromptTemplate.from_template(template)
    st.session_state.template = qa_template

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask any query"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare context (if any) and query for the prompt
    query_str = prompt
    results = vector_store.similarity_search(query_str, k=5)
    context_str = ""
    print(f"documents fetched: {len(results)}")
    for i, r in enumerate(results):
        # print(f"Result {i+1}: {r.page_content}")
        context_str += r.page_content

    prompt_text = st.session_state.template.format(context_str=context_str, query_str=query_str)
    response = llm.invoke(prompt_text)
    print(f"tool calls: {response.tool_calls}")

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response})