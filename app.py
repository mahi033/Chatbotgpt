import os
import openai
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

from flask import Flask, request, jsonify

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your chatbot index using LlamaIndex
documents = SimpleDirectoryReader('docs').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = index.query(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
