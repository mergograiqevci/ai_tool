from flask import Flask, jsonify, request
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/classify_transactions', methods=['POST'])
def classify_transactions():
    # Parse JSON data from the request body
    data = request.json
    transactions = data.get("transactions", [])
    refined_categories = data.get("categories", [])

    # Check if transactions and categories are provided
    if not transactions or not refined_categories:
        return jsonify({"error": "Transactions and categories are required"}), 400

    # Prepare the list of transaction descriptions with the format "Name: <name>, Amount: <amount>"
    transaction_descriptions = [
        f"Name: {transaction['name']}, Amount: ${transaction['amount']}" for transaction in transactions
    ]

    # Get predictions for all transaction descriptions at once
    predictions = classifier(
        transaction_descriptions,
        candidate_labels=refined_categories,
        hypothesis_template="This transaction should be categorized as {}."
    )

    # Prepare results list with predictions
    results = []
    for i, transaction in enumerate(transactions):
        result = {
            "Transaction Name": transaction["name"],
            "Amount": transaction["amount"],
            "Predicted Category": predictions["labels"][i][0],  # Get the top prediction
            "Prediction Score": predictions["scores"][i][0]  # Get the score of the top prediction
        }
        results.append(result)

    # Return results as JSON
    return jsonify(results), 200

if __name__ == '__main__':
    # Use host="0.0.0.0" to make it accessible over the network
    app.run(host="0.0.0.0", port=8080, debug=False)
