from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


# load document
def document_loader(file_path):
    loader = PyPDFLoader(file_path) if file_path.endswith(
        '.pdf') else TextLoader(file_path)
    return loader.load()


def split_documents(documents, chunk_size=800, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        collection_name="collection_legal_docs",
        embedding_function=embeddings,
        persist_directory='./chroma_legal_db'
    )
    vector_store.add_documents(chunks)
    return vector_store


resume_analysis_prompt = ChatPromptTemplate([
    ("system", """You are a professional resume analyzer and career expert. Analyze the provided resume and extract the relevant information based on the user's query.
     
    Focus on:
    - Professional experience and key responsibilities
    - Technical skills and expertise
    - Educational background and certifications
    - Projects completed and accomplishments
    - Career progression and growth
     
    Provide factual, accurate information based only on the resume content.
    """),
    ("human", "Resume text: {context} \n\n Query: {question}")
])


def create_legal_agent(file_path, query=""):
    documents = document_loader(file_path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    stuffed_chain = create_stuff_documents_chain(llm, resume_analysis_prompt)
    retrieval_chain = RunnableParallel(
        {
            'context': retriever,
            'question': RunnablePassthrough()
        }
    ) | stuffed_chain
    return retrieval_chain


def main():
    file_path = "./documents/Resume2025_2.pdf"
    query = input("Enter your query: ")
    chain = create_legal_agent(file_path)
    response = chain.invoke(query)
    print(response)


main()
