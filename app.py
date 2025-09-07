import os
import logging
from flask import Flask, render_template, request, jsonify

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Try importing retriever; if not present, fall back to get_retriever()
# and create a module-level retriever. Always import format_docs.
try:
    from vectordb import retriever, format_docs
except Exception:
    from vectordb import format_docs, get_retriever
    retriever = get_retriever()

app = Flask(__name__)

# Configure logging (avoid double-configuration when Flask reloader spawns subprocess)
# WERKZEUG_RUN_MAIN is set to "true" inside the child process that actually serves requests.
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Use Flask's logger (keeps handlers consistent with Flask)
logger = app.logger
logger.setLevel(logging.INFO)

# Initialize the model and prompt template
model = OllamaLLM(
    model="mistral",
    # These options are passed to Ollama for speed.
    options={
        "num_predict": 256,    # cap tokens generated
        "num_ctx": 2048,       # context window (keep moderate)
        "top_k": 20,
        "top_p": 0.9,
        "temperature": 0.3,
        "keep_alive": "10m",   # keep model in memory between calls
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
    """
    Call the retriever using the first available method we detect.
    Returns a list of documents.
    """
    # Common retriever method names used by different wrappers
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(question)
    if hasattr(retriever, "get_documents"):
        return retriever.get_documents(question)
    # Try a generic .retrieve or .search API if present
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
        # Accept either 'question' or older 'message' key
        question = (data.get('question') or data.get('message') or "").strip()

        if not question:
            return jsonify({'error': 'Please provide a question'}), 400

        logger.info("Processing question: %s", question)

        # Get relevant restaurant data from vector database
        docs = _retrieve_docs(question)

        # Some retriever implementations return a list of strings instead of Document objects.
        # Ensure docs is a list of Document-like objects for format_docs.
        # format_docs in vectordb expects objects with .page_content and .metadata.
        # If docs are raw strings, wrap them simply.
        if docs and isinstance(docs[0], str):
            from langchain_core.documents import Document
            docs = [Document(page_content=d, metadata={}) for d in docs]

        context = format_docs(docs, per_doc_char_limit=600)

        # Generate response using the chain
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
            'detail': str(e)  # include error string to help debugging; remove in production if desired
        }), 500

if __name__ == '__main__':
    # Optionally disable the reloader if you want only a single process (no duplicate logs):
    # app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    app.run(debug=True, host='0.0.0.0', port=5000)
