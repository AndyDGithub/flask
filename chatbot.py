from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS
import os
from huggingface_hub import login

port = int(os.environ.get("PORT", 10000))  # Render définit automatiquement le port

# Utilisez la clé API Hugging Face stockée dans la variable d'environnement
hf_token = os.getenv("ACCESS_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("Token Hugging Face non trouvé. Assurez-vous que la variable d'environnement HF_AUTH_TOKEN est définie.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Charger le modèle et le tokenizer
model_name = "facebook/blenderbot-90M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# 🔹 CORRECTION : Définir un token de padding
tokenizer.pad_token = tokenizer.eos_token

# device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

device = "cpu"  # Force le CPU si pas de GPU
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to(device)  # 🔹 Déplacer entièrement le modèle sur CPU

import traceback  # 🔹 Pour afficher les erreurs

@app.route("/chat", methods=["POST"])
def chat():
    try:
        print("🔹 Requête reçue")  # Log pour voir si la requête arrive bien

        if request.content_type != "application/json":
            return jsonify({"error": "Content-Type doit être application/json"}), 415  # 🔥 Retourner une erreur plus claire

        data = request.get_json()
        print("🔹 Données reçues:", data)

        if not data or "message" not in data:
            return jsonify({"error": "Aucun message envoyé"}), 400

        user_message = data["message"]

        # 🔹 Vérifier si le modèle est bien chargé
        if model is None:
            print("❌ Modèle non chargé")
            return jsonify({"error": "Modèle non chargé"}), 500

        inputs = tokenizer(user_message, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("🔹 Réponse générée:", bot_response)  # Log pour voir la réponse générée

        return jsonify({"response": bot_response})

    except Exception as e:
        print("❌ Erreur :", str(e))  # 🔥 Afficher l'erreur dans les logs
        traceback.print_exc()  # 🔥 Afficher la trace complète de l'erreur
        return jsonify({"error": "Erreur interne"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
