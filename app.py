from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

offload_dir = "/content/offload"
os.makedirs(offload_dir, exist_ok=True)

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

    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

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
    user_message = data.get("message", "")
    reply = generate_response(user_message)
    print("✅ Réponse générée :", reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
