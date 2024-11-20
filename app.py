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
    refined_categories = data.get("categories", {})
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Authorization token is missing or invalid"}), 401

    token = auth_header.split(" ")[1]
    user = users_collection.find_one({"jwtTokens.token": token})

    if not user:
        return jsonify({"error": "Invalid or unauthorized token"}), 403

    # Check if transactions and categories are provided
    if not transactions or not refined_categories:
        return jsonify({"error": "Transactions and categories are required"}), 400

    # Offload classification to a background thread
    def classify_and_store():
        results = []
        category_keys = list(refined_categories.keys())  # Extract category keys

        for transaction in transactions:
            transaction_name = transaction["name"]
            transaction_amount = transaction["amount"]

            # Predict the main category using transaction name and amount
            category_input = f"{transaction_name}. Amount: {transaction_amount}"
            category_prediction = classifier(
                category_input,
                candidate_labels=category_keys,
                hypothesis_template="This transaction should be categorized as {}."
            )

            predicted_category = category_prediction["labels"][0]

            # Get subcategories for the predicted category
            subcategories = refined_categories.get(predicted_category, [])

            # Predict the subcategory using transaction name, amount, and predicted category
            subcategory_input = f"{transaction_name}. Amount: {transaction_amount}. Category: {predicted_category}"
            subcategory_prediction = classifier(
                subcategory_input,
                candidate_labels=subcategories,
                hypothesis_template="This transaction belongs to {}."
            )

            predicted_subcategory = subcategory_prediction["labels"][0]

            # Update the transaction in the database
            transactions_collection.update_one(
                {
                    "transaction_id": transaction["transaction_id"],
                    "user": str(user["_id"])
                },
                {
                    "$set": {
                        "local_category": predicted_category,
                        "local_sub_category": predicted_subcategory
                    }
                }
            )

            # Append results for logging/debugging
            results.append({
                "transaction_name": transaction_name,
                "amount": transaction_amount,
                "predicted_category": predicted_category,
                "predicted_subcategory": predicted_subcategory,
                "category_score": category_prediction["scores"][0],
                "subcategory_score": subcategory_prediction["scores"][0]
            })

        print(results)  # Log the results for debugging

    thread = threading.Thread(target=classify_and_store)
    thread.start()

    return jsonify({"status": "Processing in background"}), 202

if __name__ == '__main__':
    # Use host="0.0.0.0" to make it accessible over the network
    app.run(host="0.0.0.0", port=8080, debug=False)