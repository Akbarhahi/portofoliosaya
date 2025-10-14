from flask import Flask, request, jsonify
from chatbot import Chatbot
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Path ke intents dan embeddings
script_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(script_dir, 'intents.json')
embedding_path = os.path.join(script_dir, 'intent_embeddings.pt')

# Inisialisasi chatbot (debug=False untuk produksi)
bot = Chatbot(intents_path, embedding_file=embedding_path, debug=False)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    response, _ = bot.get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
