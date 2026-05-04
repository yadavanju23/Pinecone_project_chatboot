from flask import Flask, render_template
from dotenv import load_dotenv
import os
from flask import request

# Safe imports (only import, don't execute anything heavy)
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

embeddings=download_hugging_face_embeddings()
index_name = "pinecone-research-trails"
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create the retriever and the language model
retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
chat = ChatGroq(
    model="llama-3.1-8b-instant"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain=create_stuff_documents_chain(chat,prompt)
rag_chain=create_retrieval_chain(retriever, question_answer_chain)



app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("chatboot.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")   
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response:",response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)