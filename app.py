from flask import Flask, request, jsonify
import requests
import json
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize in-memory log storage
logs = []

# Utility function to add log entries
def add_log(message, log_type="info"):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": log_type,
        "message": message
    }
    logs.append(log_entry)
    print(log_entry)  # Print to console for debugging

# AIML API Configuration
AIML_API_URL = "https://api.together.xyz/v1/chat/completions"
AIML_API_KEY = "f6d3a4ca6990e78d553a9fe773999e37f232a5726a3e5fdeae42a0032d934487"
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIML_API_KEY}'
}

# API route for performing OCR
@app.route('/api/ocr', methods=['POST'])
def perform_ocr():
    data = request.get_json()
    add_log(f"Received OCR request data: {data}")
    
    image_url = data.get('imageUrl')
    add_log(f"Image URL: {image_url}")
    
    if not image_url:
        error_msg = "Error: No image URL provided"
        add_log(error_msg, log_type="error")
        return jsonify({"error": "Image URL is required."}), 400

    try:
        payload = json.dumps({
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image, do not need any explanations just give ingredients"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 300
        })
        
        response = requests.post(AIML_API_URL, headers=HEADERS, data=payload)
        add_log(f"OCR API response: {response.status_code}")
        
        if response.status_code != 201:
            return jsonify({"error": "Failed to process OCR."}), 500
        
        result = response.json()
        extracted_text = result.get("choices", [{}])[0].get("message", {}).get("content", "No text found")
        add_log(f"OCR result: {extracted_text}")
        cleaned_text = extracted_text.replace('*', '')
        return jsonify({"text": cleaned_text})
    
    except Exception as e:
        error_msg = f"Failed to perform OCR: {str(e)}"
        add_log(error_msg, log_type="error")
        return jsonify({"error": error_msg}), 500

# API route for performing recommendation
@app.route('/api/recommend', methods=['POST'])
def perform_recommend():
    data = request.get_json()
    add_log(f"Received recommendation request data: {data}")
    
    patient_data = data.get('patient')
    ingredients = data.get('ingredients')
    add_log(f"Patient data: {patient_data}, Ingredients: {ingredients}")

    if not patient_data or not ingredients:
        add_log("Error: Missing patient data or ingredients", log_type="error")
        return jsonify({"error": "Missing patient data or ingredients"}), 400

    prompt = f"""
    Patient Information:
    - Age: {patient_data.get('age')}
    - Allergies: {patient_data.get('allergies')}
    - Health Conditions: {patient_data.get('healthConditions')}
    - Sugar Level: {patient_data.get('sugarLevel')}
    - Blood Pressure: {patient_data.get('bloodPressure')}
    
    Ingredients:
    {ingredients}
    Based on the provided ingredients assume person is consuming 100gms or 1 serving, provide a nutritional breakdown including:
    - Total Calories
    - Percentage of Protein, Carbohydrates, and Fats
    - Total Fat in grams and percentage of daily value
    - Saturated Fat in grams and percentage of daily value
    - Trans Fat in grams
    - Any Ammino acids or Acid Content or Organic Acids
    Afterward, provide a recommendation on whether this food is safe for the patient to consume, considering their health profile.
    """
    add_log(f"Generated prompt for AIML API: {prompt}")

    try:
        payload = json.dumps({
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        })

        response = requests.post(AIML_API_URL, headers=HEADERS, data=payload)
        add_log(f"Recommendation API response: {response.status_code}")

        if response.status_code != 201:
            return jsonify({"error": "Failed to process recommendation."}), 500

        result = response.json()
        recommendation = result.get("choices", [{}])[0].get("message", {}).get("content", "No recommendation available")
        add_log(f"Recommendation result: {recommendation.replace('*', '')}")
        print(recommendation)

        p2 = """
            Extract the following nutritional values from the given text and provide single representative values in JSON format:

            {
                "calories": <single average value in kcal>,
                "protein": <single average value in grams>,
                "carbs": <single average value in grams>,
                "fat": <single average value in grams>,
                "detailed_nutrition": {
                    "total_fat": <single average value in grams>,
                    "saturated_fat": <single average value in grams>,
                    "trans_fat": <single average value in grams>
                },
                "graph": {
                    "protein": <single average value in percentage>,
                    "carbs": <single average value in percentage>,
                    "fat": <single average value in percentage>
                }
            }

            Ensure that the extracted values are accurate and formatted correctly as JSON, no need any extra information, only json is enough.
            """  + recommendation
        
        pay2 = json.dumps({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": p2}],
            "max_tokens": 1024
        })

        response = requests.post(AIML_API_URL, headers=HEADERS, data=pay2)
        add_log(f"Recommendation API response: {response.status_code}")

        if response.status_code != 201:
            return jsonify({"error": "Failed to process recommendation."}), 500
        
        result = response.json()
        nutrition_data = result.get("choices", [{}])[0].get("message", {}).get("content", "No recommendation available")
        add_log(f"nutrition_data result: {nutrition_data}")
        nutrition=nutrition_data.replace('*', '').replace("```", '').replace("json",'')
        nutrition_json = json.loads(nutrition)
        print(nutrition_json)
        add_log(f"Parsed nutrition data: {nutrition_json}")

        return jsonify({
            "recommendation": recommendation.replace('*', ''),
            "nutrition": nutrition_json
        })

    except Exception as e:
        error_msg = f"Failed to perform recommendation: {str(e)}"
        add_log(error_msg, log_type="error")
        return jsonify({"error": error_msg}), 500

# API route for logs
@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": logs})

@app.route('/', methods=['GET'])
def hello():
    add_log("Hello endpoint accessed")
    return "Hello"

# Start the Flask server
if __name__ == '__main__':
    add_log("Starting Flask server...")
    app.run(host="0.0.0.0", port=3009, debug=True)
