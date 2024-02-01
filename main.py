import streamlit as st
import openai
import langchain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone 
import pinecone 

import os
os.environ['OPENAI_API_KEY']=st.secrets["OPENAI_API_KEY"]
os.environ['PINECONE_API_KEY']=st.secrets["PINECONE_API_KEY"] 

## Streamlit UI
st.set_page_config(page_title="Q&A Chatbot")
st.header("The 48 Laws Of Power")

def get_vector_store(index_name="rag"):
  embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
  #setting up the piinecone
  pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])
  index =pc.Index(index_name)
  # docsearch = Pinecone.from_texts(chunks,embeddings, index_name="rag")
  docsearch = Pinecone.from_existing_index(index_name, embeddings)
  return docsearch

docsearch = get_vector_store()

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model=OpenAI(model_name="gpt-3.5-turbo",temperature=0.7)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def get_response(user_question):
  docs = docsearch.similarity_search(user_question,k=3)
  chain = get_conversational_chain()
  response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
  return response['output_text']

input=st.text_input("Ask the Question")
response=get_response(input)

submit=st.button("submit")

if submit:
    st.write(response)
