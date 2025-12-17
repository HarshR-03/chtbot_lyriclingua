import faiss
import os
import pickle
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dir = "./embeddings"

def load_vector_store(index_name, embedding_model):
    index = faiss.read_index(os.path.join(embedding_dir, f"{index_name}.faiss"))
    with open(os.path.join(embedding_dir, f"{index_name}_docstore.pkl"), "rb") as f:
        docstore = pickle.load(f)
    with open(os.path.join(embedding_dir, f"{index_name}_idmap.pkl"), "rb") as f:
        id_map = pickle.load(f)
    return FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=id_map,
    )

def combine_vector_stores(stores):
    combined_store = stores[0]
    current_index_size = combined_store.index.ntotal

    for store in stores[1:]:
        # Merge FAISS index
        combined_store.index.merge_from(store.index)
        # Merge docstore
        combined_store.docstore._dict.update(store.docstore._dict)
        # Shift the indices of the second store's id map
        shifted_id_map = {
            k + current_index_size: v for k, v in store.index_to_docstore_id.items()
        }
        combined_store.index_to_docstore_id.update(shifted_id_map)
        current_index_size += store.index.ntotal

    return combined_store

if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store1 = load_vector_store("grammar1", embedding_model)
    store2 = load_vector_store("grammar2", embedding_model)
    combined_store = combine_vector_stores([store1, store2])

    # Query the combined store
    query = "Explain the difference between wa and ga in Japanese."
    results = combined_store.similarity_search(query, k=5)
    for i, r in enumerate(results):
        print(f"Result {i+1}: {r.page_content}")

    # Pass results to your LLM (pseudo-code)
    # response = your_llm_chain_here(query=query, context=[r.page_content for r in results])
    # print(response)
