import os
import logging
from flask import Flask, render_template, request, jsonify

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

try:
    from vectordb import retriever, format_docs
except Exception:
    from vectordb import format_docs, get_retriever
    retriever = get_retriever()

app = Flask(__name__)

if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = app.logger
logger.setLevel(logging.INFO)

model = OllamaLLM(
    model="mistral",
    options={
        "num_predict": 256,
        "num_ctx": 2048,
        "top_k": 20,
        "top_p": 0.9,
        "temperature": 0.3,
        "keep_alive": "10m",
    },
)

template = """
You are a helpful restaurant concierge and dining expert.

Use ONLY the restaurant details in the context to produce tailored recommendations, adding brief expert color if reasonable.
Prefer factual fields from context over guesses.

Do not mention that you are using retrieved/provided data. Just provide the best possible answer.

Return:
1) A succinct summary of who these places are best for.
2) 3-4 recommendations as bullets; each with:
   - Name
   - Why it matches the request (cuisine, price, vibe, dietary fit)
   - Key details (Name, Cleanliness, Service, Pricing, Food/Drinks, Ambience, Overall, What to Try, Price per head out of 5)
   - Also mention the Location if present.
3) Keep the data presented structured and DO NOT mention any numeric bullet points.

MENTION ONLY THE DETAILS PRESENT IN THE CONTEXT. DO NOT MAKE UP DETAILS.

Restaurant Data:
{context}

User Question:
{question}

Final Answer: (only bullets and a little information)

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def _retrieve_docs(question):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)
    if hasattr(retriever, "get_documents"):
        return retriever.get_documents(question)
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(question)
    if hasattr(retriever, "search"):
        return retriever.search(question)
    raise RuntimeError("Could not call retriever: no supported method found (invoke/get_relevant_documents/get_documents/retrieve/search).")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = (data.get('question') or data.get('message') or "").strip()

        if not question:
            return jsonify({'error': 'Please provide a question'}), 400

        logger.info("Processing question: %s", question)

        docs = _retrieve_docs(question)

        if docs and isinstance(docs[0], str):
            from langchain_core.documents import Document
            docs = [Document(page_content=d, metadata={}) for d in docs]

        context = format_docs(docs, per_doc_char_limit=600)

        result = chain.invoke({"context": context, "question": question})

        logger.info("Response generated successfully")
        return jsonify({
            'response': result,
            'status': 'success'
        })

    except Exception as e:
        logger.exception("Error processing request")
        return jsonify({
            'error': 'Sorry, there was an error processing your question. Please try again.',
            'status': 'error',
            'detail': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
