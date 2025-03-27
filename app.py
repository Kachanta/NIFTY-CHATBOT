import streamlit as st
from langchain_core.runnables.passthrough import RunnablePassthrough

# Ensure required dependencies are installed
try:
    import streamlit
    import pinecone
except ModuleNotFoundError as e:
    st.error(f"Missing module: {e.name}. Please install it using 'pip install {e.name}'")
    st.stop()

# Import your RAG components
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
pinecone_key=st.secrets["PINECONE_API_KEY"]
# Initialize Pinecone
if not pinecone_key:
    st.error("Pinecone API key is missing. Set it as an environment variable.")
    st.stop()

pc = Pinecone()
index_name = "final-dsfp-bot"  
index = pc.Index(index_name)

# Initialize NVIDIA embeddings
embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")

# Initialize Pinecone Vector Store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(k=1)

cohere_api=st.secrets["COHERE_API_KEY"]
# Initialize LLM
if not cohere_api:
    st.error("Cohere API key is missing. Set it as an environment variable.")
    st.stop()

cohere_llm = ChatCohere(cohere_api_key=cohere_api)

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    "You are a knowledgeable AI assistant. Answer the question based only on the provided context.\n\n"
    "Context: {context}\n"
    "Question: {question}\n\n"
    "If the context does not contain relevant information, say: 'I don/’t know based on the given context.'"
    "Give a concise answer only without including things like content= or page_content= or any other metadata."
)

# Function to convert retrieved documents to string
def doc2str(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Define RAG Chain
# rag_chain = (
#     {"context": retriever | doc2str, "question": RunnablePassthrough()}
#     | prompt
#     | cohere_llm
# )

# Define a function to process user queries
def process_query(query):
    retrieved_docs = retriever.invoke(query)  # Retrieve relevant documents
    context = doc2str(retrieved_docs)  # Convert documents to string format
    formatted_prompt = prompt.invoke({"context": context, "question": query})  # Format the prompt
    response = cohere_llm.invoke(formatted_prompt)  # Generate response using LLM
    return response

# Example usage
#response = process_query("What is NFTI?")


# Streamlit UI
# st.title("NFTI CHATBOT 🤖")
# st.write("Chat with NFTI BOT to learn more about NFTI!")

st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 150px;">
        <h1>NFTI CHATBOT</h1>
        <p>Ask me anything about NFTI. I'll provide an answer.</p>
    </div>
""", unsafe_allow_html=True)


# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Input
user_query = st.text_input("Ask me anything about NFTI")
if st.button("Submit") and user_query:
    with st.chat_message("user"):
        st.write(user_query)    
    # Get Response
    response = process_query(user_query)

    with st.chat_message("assistant"):
          st.markdown(f"<p style='font-size:14pt; font-weight:'';'>{response.content}</p>", unsafe_allow_html=True)

    # Store conversation history
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.session_state["messages"].append({"role": "assistant", "content": response})
