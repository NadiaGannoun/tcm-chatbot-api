from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from flask_cors import CORS

offload_dir = "/content/offload"
os.makedirs(offload_dir, exist_ok=True)

ngrok.set_auth_token("30SkcewFprCQobQqAEeRvYHPx6R_21fZkyVtGCAeBKRekKtXn")

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

model_path = "/content/drive/MyDrive/results/checkpoint-1690"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder=offload_dir
)

def generate_response(message):
    prompt = f"[INST] <<sys>>\nHi! I am your AI assistant specializing in Traditional Chinese Medicine.\n<</sys>>\n\n{message} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.95, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    if '[/INST]' in response:
        return response.split('[/INST]')[-1].strip()
    else:
        return response.strip()

@app.route('/', methods=['GET'])
def home():
    return "TraditionalCareGPT backend is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print("✅ Requête reçue :", data)
    user_msg = data.get("message", "")
    reply = generate_response(user_msg)
    print("✅ Réponse générée :", reply)
    return jsonify({"reply": reply})

def run():
    app.run(port=5000)

threading.Thread(target=run).start()
public_url = ngrok.connect(5000)
print(" URL publique :", public_url)
