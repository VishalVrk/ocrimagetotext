from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the Gradio clients
ocr_client = Client("vrkforever/OCR-image-to-text")
bot_client = Client("vrkforever/Chat-Llama-3.2-3B-Instruct-uncensored")

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
        response = requests.get(image_url)
        add_log(f"Image fetch response status code: {response.status_code}")

        if response.status_code != 200:
            error_msg = "Error: Failed to fetch image from URL"
            add_log(error_msg, log_type="error")
            return jsonify({"error": error_msg}), 400

        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        img.save("temp_image.png")
        add_log("Image successfully saved as temp_image.png")

        result = ocr_client.predict(
            Method="PaddleOCR",
            img=handle_file("temp_image.png"),
            api_name="/predict"
        )
        add_log(f"OCR result: {result}")

        bot_result = bot_client.predict(
            message=f"{result} \n based on the given value keep only ingredients. If no ingredients are found, say 'No Ingredients found'",
            system_prompt=f"{result} \n based on the given value keep only ingredients. If no ingredients are found, say 'No Ingredients found'",
            max_new_tokens=1024,
            temperature=0.6,
            api_name="/chat"
        )
        add_log(f"Bot client result: {bot_result}")

        clean_output = bot_result.replace("assistant", "").strip()
        ingredients_list = clean_output.split("\n")
        ingredients_list = [ingredient.split(". ", 1)[1] if ". " in ingredient else ingredient for ingredient in ingredients_list]
        add_log(f"Extracted ingredients list: {ingredients_list}")

        return jsonify(ingredients_list)

    except Exception as e:
        error_msg = f"Failed to perform OCR: {str(e)}"
        add_log(error_msg, log_type="error")
        return jsonify({"error": error_msg}), 500


@app.route('/api/recommend', methods=['POST'])
def perform_recomm():
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
    Afterward, provide a recommendation on whether this food is safe for the patient to consume, considering their health profile.
    """
    add_log(f"Generated prompt for bot client: {prompt}")

    try:
        bot_result = bot_client.predict(
            message=prompt,
            system_prompt=prompt,
            max_new_tokens=1024,
            temperature=0.6,
            api_name="/chat"
        )
        add_log(f"Bot client response: {bot_result}")

        clean_output = bot_result.replace("assistant", "").replace("**","").strip()
        add_log(f"Cleaned output: {clean_output}")

        nutrition_data = {
            "calories": None,
            "protein": None,
            "carbs": None,
            "fat": None,
            "detailed_nutrition": {
                "total_fat": None,
                "saturated_fat": None,
                "trans_fat": None
            },
            "graph": {
                "protein": None,
                "carbs": None,
                "fat": None
            }
        }

        for line in clean_output.splitlines():
            if "Calories" in line:
                nutrition_data["calories"] = line.split(":")[1].strip()
            elif "Protein" in line:
                nutrition_data["protein"] = line.split(":")[1].strip()
                nutrition_data["graph"]["protein"] = nutrition_data["protein"][nutrition_data["protein"].find("(")+1:nutrition_data["protein"].find("%")]
            elif "Carbohydrates" in line:
                nutrition_data["carbs"] = line.split(":")[1].strip()
                nutrition_data["graph"]["carbs"] = nutrition_data["carbs"][nutrition_data["carbs"].find("(")+1:nutrition_data["carbs"].find("%")]
            elif "Fat:" in line:
                nutrition_data["fat"] = line.split(":")[1].strip()
                nutrition_data["detailed_nutrition"]["total_fat"] = line.split(":")[1].strip()
                nutrition_data["graph"]["fat"] = nutrition_data["fat"][nutrition_data["fat"].find("(")+1:nutrition_data["fat"].find("%")]
            elif "Saturated Fat" in line:
                nutrition_data["detailed_nutrition"]["saturated_fat"] = line.split(":")[1].strip()
            elif "Trans Fat" in line:
                nutrition_data["detailed_nutrition"]["trans_fat"] = line.split(":")[1].strip()

        add_log(f"Parsed nutrition data: {nutrition_data}")

        return jsonify({
            "recommendation": clean_output,
            "nutrition": nutrition_data
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