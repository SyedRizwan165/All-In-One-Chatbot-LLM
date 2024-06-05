# Import necessary libraries
import streamlit as st
import json
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI 
from langchain.chains import ConversationalRetrievalChain
import openai
from openai.embeddings_utils import get_embedding

# Set your OpenAI API key
OPENAI_API_KEY = ""

# Set up Streamlit app title
st.title("Chat With Your Document 📑")

# Allow user to upload a document
uploaded_doc = st.file_uploader("Choose a file")

# Process the uploaded document
if uploaded_doc:
    # Check the type of the document and extract text accordingly
    if uploaded_doc.type == "application/pdf":
        doc_reader = PdfReader(uploaded_doc)
        raw_text = ""
        for page in doc_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
    elif uploaded_doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        word_file = docx.Document(uploaded_doc)
        raw_text = ' '.join([para.text for para in word_file.paragraphs])
    else:
        # Display an error for unsupported formats
        st.error("Unsupported format")
        st.stop()
    
    # Split the document into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = splitter.split_text(raw_text)

    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Set up FAISS for similarity search
    similarity_search = FAISS.from_texts(texts, embeddings)

    # Set up ConversationalRetrievalChain for question answering
    chain_qa= ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        retriever=similarity_search.as_retriever(),
        return_source_documents=True,
    )

    # Initialize chat history
    chat_history = []

    # Allow the user to input a question
    query = st.text_input("Ask a question about the uploaded document:")

    # Generate the answer when the button is clicked
    if st.button("Generate Answer"):
        result = chain_qa({"question": query, "chat_history": chat_history})
        st.success(result["answer"])

# in order to generate and display embeddings
show_embeddings = False  
generate_embeddings_button = st.button("Generate Embeddings")
if generate_embeddings_button:
    show_embeddings = True

@st.cache_data
def generate_embeddings():
    # Get query embedding
    query_embedding = get_embedding(query, engine="text-embedding-ada-002", api_key=OPENAI_API_KEY)
    query_embedding_str = json.dumps(query_embedding)

    # Get document embeddings
    doc_embeddings = []
    for doc_text in texts:
        doc_embed = get_embedding(doc_text, engine="text-embedding-ada-002", api_key=OPENAI_API_KEY)
        doc_embeddings.append(doc_embed)

    # Get similar docs 
    similar_docs = similarity_search.search(query_embedding_str, k=5, search_type="similarity")

    return query_embedding, doc_embeddings, similar_docs

# Display embeddings if the button is clicked
if show_embeddings:
    query_embedding, doc_embeddings, similar_docs = generate_embeddings()
    # Show query embeddings
    st.subheader("Query Embedding:")
    st.write(query_embedding)
    # Show document embeddings
    st.subheader("Document Embeddings:")
    for i, doc_embed in enumerate(doc_embeddings):
        st.write(f"Document {i+1} Embedding:")
        st.write(doc_embed)
    # Show similar document embeddings
    st.subheader("Similar Documents:")
    st.write(similar_docs)




