import base64
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from utils import load_and_split_pdf_document, get_vector_store, get_context_aware_retriever_chain,  get_conversational_rag_chain


from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Retrieval Augmented Generation", page_icon="🤖")
st.title("Chat with PDF")

with st.sidebar:
    st.header("Settings")
    pdf = st.file_uploader("Please upload your PDF", type="pdf")
    

if pdf is None:
    st.info("Please upload PDF!")
else:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    base64_pdf = base64.b64encode(pdf.read()).decode("utf-8")
    pdf_display = (
            f'<embed src="data:application/pdf;base64,{base64_pdf}#zoom=40" '
            'width="400" height="500" type="application/pdf"></embed>'
        )
    
    with st.sidebar:    
        st.markdown(pdf_display, unsafe_allow_html=True)                                

    document_chunks = load_and_split_pdf_document(pdf)
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