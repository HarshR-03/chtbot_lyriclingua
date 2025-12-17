from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss
from dotenv import load_dotenv
import os
import pickle
load_dotenv()

data_base_dir = "./data/"
embedding_dir = "./embeddings"

def load_index(paths: list[str], index_name:str, page_slice_indices: list[int]):
    # pages = []
    loader = PyPDFLoader(data_base_dir + paths[0])
    docs = loader.load()
    if(page_slice_indices):
        # ensure correct slices
        l,r = page_slice_indices
        docs = docs[l:r]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if(not os.path.exists(embedding_dir)):
        os.makedirs(embedding_dir)

    # test = embedding_model.embed_documents("konnichiwa")
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("konnichiwa")))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(all_splits)

    faiss.write_index(vector_store.index, os.path.join(embedding_dir, f"{index_name}.faiss"))
    with open(os.path.join(embedding_dir, f"{index_name}_docstore.pkl"), "wb") as f:
        pickle.dump(vector_store.docstore, f)
    with open(os.path.join(embedding_dir, f"{index_name}_idmap.pkl"), "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)

    # print("embedding: ", test)
    print(f"no. of pages: {len(docs)}")
    print(f"page 1: {docs[2]}")
    print(f"splits length: {len(all_splits)}")

if __name__ == "__main__":
    load_index(["Japanese Grammar, Modern_ a Practical Guide.pdf"],"grammar1",[22,418])
    load_index(["Japanese, Tae Kim's Grammar Guide.pdf"],"grammar2",[11,352])

