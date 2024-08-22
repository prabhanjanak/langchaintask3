import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from googleapiclient.discovery import build

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Streamlit interface
st.image("logo.png", width=150)  # Add logo here
st.title("Resolute AI - Task 3")
st.write("Interactive Document Query System with RAG and Google Gemini Integration - AUG2024")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with open(f"uploaded_file.{file_extension}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if file_extension == "pdf":
        loader = PyPDFLoader(f"uploaded_file.{file_extension}")
    elif file_extension == "docx":
        loader = Docx2txtLoader(f"uploaded_file.{file_extension}")
    elif file_extension == "txt":
        loader = TextLoader(f"uploaded_file.{file_extension}")
    else:
        st.error("Unsupported file type")
        st.stop()

    documents = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Store the retriever in the session state
    st.session_state["retriever"] = retriever
    st.session_state["messages"].append({"role": "system", "content": "File processed successfully. You can now ask questions."})

# Display chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f'**You:** {message["content"]}')
    else:
        st.markdown(f'**Bot:** {message["content"]}')

# Query input
query = st.text_input("Enter your query here")

if query and "retriever" in st.session_state:
    retriever = st.session_state["retriever"]
    
    # Load the LLM
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_new_tokens=1024, temperature=1, huggingfacehub_api_token=st.secrets['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']
    )
    
    # Create the QA chain
    qa_chain_with_sources = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        callbacks=[StdOutCallbackHandler()],
        return_source_documents=True
    )
    
    # Get the response from the RAG model
    response = qa_chain_with_sources({"query": query})
    result = response["result"]
    
    # Store user query and bot response in session state
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["messages"].append({"role": "bot", "content": result})
    
    # Optionally display the source documents
    source_docs = "\n".join([doc.page_content for doc in response["source_documents"]])
    
    # Integrate Google Gemini API for enhanced interactions
    api_key = st.secrets['AIzaSyCwoOEEJvucaPxeil3jOGET0KnmmrJ-cJA']
    google_gemini_service = build('gemini', 'v1', developerKey=api_key)
    
    try:
        gemini_response = google_gemini_service.text().analyze(
            text=query
        ).execute()
        
        # Display the response from Google Gemini
        gemini_analysis = gemini_response.get('analysis', 'No analysis available')
        st.session_state["messages"].append({"role": "bot", "content": f"Google Gemini Analysis: {gemini_analysis}"})
    except Exception as e:
        st.session_state["messages"].append({"role": "bot", "content": f"Error interacting with Google Gemini: {e}"})

# Display updated chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f'**You:** {message["content"]}')
    else:
        st.markdown(f'**Bot:** {message["content"]}')
