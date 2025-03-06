from flask import Flask, request, jsonify
from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://portfolio-mu-steel-62.vercel.app/"]) # Permettre les requêtes cross-origin

# Charger le modèle et le tokenizer
model_name = "./OpenHermes-2.5-Mistral-7B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 🔹 CORRECTION : Définir un token de padding
tokenizer.pad_token = tokenizer.eos_token

# device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

device = "cpu"  # ⚠️ Force le CPU si pas de GPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to(device)  # 🔹 Déplacer entièrement le modèle sur CPU


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Aucun message envoyé"}), 400

        user_message = data["message"]

        # 🔹 CORRECTION : Ajouter padding et truncation
        inputs = tokenizer(user_message, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id  # 🔹 CORRECTION : Utiliser le pad_token_id défini plus haut
            )

        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
