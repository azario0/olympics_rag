from flask import Flask, render_template, request, jsonify
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


app = Flask(__name__)

# Initialize the RAG system
GOOGLE_API_KEY = "YOUR_API_KEY"

def setup_retrieval_qa(llm, vector_store) -> RetrievalQA:
    prompt_template = """You are an Olympics data expert. Use the following contextual information about Olympic medals, 
    GDP, and population to answer the question. Be precise with numbers and always mention the source country when relevant.
    If you don't have enough information to answer accurately, please say so.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

def query_olympics(llm, vector_store, question: str) -> Dict:
    if not vector_store:
        raise ValueError("Vector store not initialized. Please create or load a vector store first.")
    qa_chain = setup_retrieval_qa(llm, vector_store)
    result = qa_chain.invoke({"query": question})
    return {"answer": result["result"], "source_documents": result["source_documents"]}

# Initialize embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_query"
)

llm = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    top_p=0.9,
    top_k=40,
    max_output_tokens=2048,
)

# Load the vector store with allow_dangerous_deserialization=True
vector_store = FAISS.load_local(
    "olympics_vector_store", 
    embeddings,
    allow_dangerous_deserialization=True  # Added this parameter
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        response = query_olympics(llm, vector_store, question)
        
        sources = []
        for doc in response['source_documents']:
            sources.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })

        return jsonify({
            'answer': response['answer'],
            'sources': sources
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)