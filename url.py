import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from utils import load_and_split_URL, get_vector_store, get_context_aware_retriever_chain,  get_conversational_rag_chain


from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="RAG - URL", page_icon="🤖")
st.title("Chat with URL")

with st.sidebar:
    st.header("Enter URL to chat with")
    web_page_url = st.text_input("Please enter any webpage url")
    

if web_page_url is None or web_page_url == "":
    st.info("Please enter URL")
else:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    document_chunks = load_and_split_URL(web_page_url)
    vector_store = get_vector_store(document_chunks)
    retriever_chain = get_context_aware_retriever_chain(vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = vector_store
    
    user_query = st.chat_input("Type your message here...")
    
    with st.spinner():
        if user_query is not None and user_query != "":
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))

        
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)