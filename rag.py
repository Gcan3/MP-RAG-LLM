# import app and rag needed libraries
import os
import streamlit as st
import sqlite3
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Simple UI for streamlit
st.set_page_config(page_title="Microplastic QA", page_icon="🔍", layout="wide")
st.title("Microplastic QA")

# load API key from .env file
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets['HUGGINGFACEHUB_API_TOKEN']

# Template for the queries
system_prompt = """
You are given a research document for context.
You must answer the question given to you based on the context provided. Do not use any external resources.
If you do not know the answer, please respond with "I don't know". Use three sentences maximum.
/n
Context: {context}
"""
# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

# Call the embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# Get the vector storage
load_vector_store = Chroma(persist_directory="storage/microplastic_cosine", embedding_function=embeddings)

# Load the retriever
retriever = load_vector_store.as_retriever(search_kwargs={"k": 2})

# Load the LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03
)

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

def answer(question):
    response = rag_chain.invoke({'input': question})
    full_ans = response['answer']
    return full_ans

# Main function for the whole process with streamlit
def main():
    st.text('This is a simple RAG application with containing a pdf data based on a research on toxic effect of microplastics on terrestrial and aquatic plants. Check more here: https://www.sciencedirect.com/science/article/abs/pii/S0048969721034045')
    st.text('Please enter your question in the text box below and click on the "Generate Response" button to get the answer.')
    # Ask the user for the question
    text_query = st.text_input("Enter your question here:", placeholder="Why is microplastic harmful for aquatic plants?")
    
    generate_response = st.button("Generate Response")
    
    st.subheader("Response:")
    if generate_response and text_query:
        with st.spinner("Generating response..."):
            response = answer(text_query)
            if response:
                st.write(response)
                st.success("Response generated successfully.")
            else:
                st.error("No response generated.")

if __name__ == "__main__":
    main()
