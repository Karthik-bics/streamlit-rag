import tempfile
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI





def split_documents(document):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, separators="")
    document_chunks = splitter.split_documents(document)
    return document_chunks


def load_and_split_pdf_document(pdf):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf.getvalue())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    document = loader.load()
    document_chunks=split_documents(document)
    return document_chunks


def load_and_split_URL(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    document_chunks=split_documents(document)
    return document_chunks


def get_vector_store(document_chunks):
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


def get_context_aware_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0.3)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retrieval_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retrieval_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)
    