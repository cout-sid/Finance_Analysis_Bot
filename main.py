import os
import streamlit as st
import pickle
import faiss
import time
# from langchain import OpenAI  --> deprecated
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()  

st.title("FinBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    # vectorstore_openai = FAISS.from_documents(docs, embeddings)
    db = FAISS.from_documents(docs, embeddings)

    pkl = db.serialize_to_bytes()

    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

    

    

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)

    # # Assuming vectorstore_openai is a FAISS index
    # original_index = vectorstore_openai

    # # Create a new FAISS index with the same parameters
    # new_index = faiss.index_factory(original_index.d, "Flat")  # You may need to adjust the index type based on your use case

    # # Add the vectors to the new index
    # new_index.add(original_index.reconstruct_n(0, original_index.ntotal))

    # # Remove the threading lock (RLock) attribute
    # if hasattr(new_index, 'lock'):
    #     new_index.lock = None


    # # Save the FAISS index to a pickle file
    # with open(file_path, "wb") as f:
    #     pickle.dump(new_index, f)

    # db.save_local("faiss_index")

    # new_db = FAISS.load_local("faiss_index", embeddings)

    
    
    

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            embeddings = OpenAIEmbeddings()
            new_db = FAISS.deserialize_from_bytes(
            embeddings=embeddings, serialized=vectorstore
            )
            
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=new_db.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)




# https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
# https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
# https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html
# Apart from Tata Motors and Mahindra,  how many other comapnies have applied for certification