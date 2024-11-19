import threading
import os
from flask import Flask, jsonify, request
from transformers import pipeline
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
MONGO_URI = os.getenv("MONGO_URI")
print(MONGO_URI)
client = MongoClient(MONGO_URI)
db = client["admin"]  # Replace with your database name
users_collection = db["users"]  # Replace with your collection name
transactions_collection = db["transactions"]

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/classify_transactions', methods=['POST'])
def classify_transactions():
    # Parse JSON data from the request body
    data = request.json
    transactions = data.get("transactions", [])
    refined_categories = data.get("categories", [])
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Authorization token is missing or invalid"}), 401

    token = auth_header.split(" ")[1]
    print(auth_header)
    user = users_collection.find_one({"tokens.token": token})
    print(user)

    if not user:
        return jsonify({"error": "Invalid or unauthorized token"}), 403

    # Check if transactions and categories are provided
    if not transactions or not refined_categories:
        return jsonify({"error": "Transactions and categories are required"}), 400

    # Offload classification to a background thread
    def classify_and_store():
        results = []
        for transaction in transactions:
            prediction = classifier(
                transaction["name"],
                candidate_labels=refined_categories,
                hypothesis_template="This transaction should be categorized as {}."
            )

            predicted_category = prediction["labels"][0]
            predicted_score = prediction["scores"][0]
            transactions_collection.update_one(
                {
                    "_id": transaction["_id"],  # Match the transaction by its unique ID
                    "user": user["_id"]  # Ensure the transaction belongs to the authenticated user
                },
                {
                    "$set": {"local_category": predicted_category}
                }
            )

            results.append({
                "transaction_name": transaction["name"],
                "amount": transaction["amount"],
                "predicted_category": predicted_category,
                "prediction_score": predicted_score
            })
        print(results)  # Simulate storing or processing results

    thread = threading.Thread(target=classify_and_store)
    thread.start()

    return jsonify({"status": "Processing in background"}), 202

if __name__ == '__main__':
    # Use host="0.0.0.0" to make it accessible over the network
    app.run(host="0.0.0.0", port=8080, debug=False)