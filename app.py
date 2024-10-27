from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import requests
from io import BytesIO
from PIL import Image
import re
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Gradio clients
ocr_client = Client("vrkforever/OCR-image-to-text")
bot_client = Client("chuanli11/Chat-Llama-3.2-3B-Instruct-uncensored")

# API route for performing OCR
@app.route('/api/ocr', methods=['POST'])
def perform_ocr():
    data = request.get_json()
    print("Received OCR request data:", data)

    image_url = data.get('imageUrl')
    print("Image URL:", image_url)
    
    if not image_url:
        print("Error: No image URL provided")
        return jsonify({"error": "Image URL is required."}), 400

    try:
        # Fetch the image from the provided URL
        response = requests.get(image_url)
        print("Image fetch response status code:", response.status_code)

        if response.status_code != 200:
            print("Error: Failed to fetch image from URL")
            return jsonify({"error": "Failed to fetch image from URL."}), 400

        # Convert the image to a file-like object for Gradio
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        print("Image successfully opened")

        # Save image temporarily in memory for Gradio
        img.save("temp_image.png")
        print("Image saved as temp_image.png")

        # Perform OCR using the Gradio client
        result = ocr_client.predict(
            Method="PaddleOCR",
            img=handle_file("temp_image.png"),
            api_name="/predict"
        )
        print("OCR result:", result)

        # Prepare the input for bot_client by ensuring the OCR result is correctly inserted into the message
        bot_result = bot_client.predict(
            message=f"{result} \n based on the given value keep only ingredients. If no ingredients are found, say 'No Ingredients found'",
            system_prompt=f"{result} \n based on the given value keep only ingredients. If no ingredients are found, say 'No Ingredients found'",
            max_new_tokens=1024,
            temperature=0.6,
            api_name="/chat"
        )
        print("Bot client result:", bot_result)

        clean_output = bot_result.replace("assistant", "").strip()
        ingredients_list = clean_output.split("\n")
        ingredients_list = [ingredient.split(". ", 1)[1] if ". " in ingredient else ingredient for ingredient in ingredients_list]
        print("Extracted ingredients list:", ingredients_list)

        json_response = jsonify(ingredients_list)
        print("JSON response prepared")

        # Return the OCR result as a JSON response
        return json_response

    except Exception as e:
        print("Error performing OCR:", e)
        return jsonify({"error": "Failed to perform OCR", "details": str(e)}), 500

# API route for performing recommendation
@app.route('/api/recommend', methods=['POST'])
def perform_recomm():
    data = request.get_json()
    print("Received recommendation request data:", data)

    patient_data = data.get('patient')
    ingredients = data.get('ingredients')
    print("Patient data:", patient_data)
    print("Ingredients:", ingredients)

    if not patient_data or not ingredients:
        print("Error: Missing patient data or ingredients")
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
    print("Generated prompt for bot client:", prompt)

    try:
        bot_result = bot_client.predict(
            message=prompt,
            system_prompt=prompt,
            max_new_tokens=1024,
            temperature=0.6,
            api_name="/chat"
        )
        print("Bot client result:", bot_result)

        clean_output = bot_result.replace("assistant", "").replace("**","").strip()
        print("Cleaned output:", clean_output)

        # Initialize nutrition data structure
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
            "graph":{
                "protein": None,
                "carbs": None,
                "fat": None,
            }
        }

        # Extract nutritional information from the bot's response
        for line in clean_output.splitlines():
            print("Processing line:", line)
            if "Calories" in line:
                nutrition_data["calories"] = line.split(":")[1].strip()
            elif "Protein" in line:
                nutrition_data["protein"] = line.split(":")[1].strip()
                nutrition_data["graph"]["protein"]= nutrition_data["protein"][nutrition_data["protein"].find("(")+1:nutrition_data["protein"].find("%")]
            elif "Carbohydrates" in line:
                nutrition_data["carbs"] = line.split(":")[1].strip()
                nutrition_data["graph"]["carbs"]= nutrition_data["carbs"][nutrition_data["carbs"].find("(")+1:nutrition_data["carbs"].find("%")]
            elif "Total Fat" in line:
                nutrition_data["fat"] = line.split(":")[1].strip()
                nutrition_data["detailed_nutrition"]["total_fat"] = line.split(":")[1].strip()
                nutrition_data["graph"]["fat"]= nutrition_data["fat"][nutrition_data["fat"].find("(")+1:nutrition_data["fat"].find("%")]
            elif "Saturated Fat" in line:
                nutrition_data["detailed_nutrition"]["saturated_fat"] = line.split(":")[1].strip()
            elif "Trans Fat" in line:
                nutrition_data["detailed_nutrition"]["trans_fat"] = line.split(":")[1].strip()

        print("Extracted nutrition data:", nutrition_data)

        # Return the recommendation along with detailed nutrition data
        return jsonify({
            "recommendation": clean_output,
            "nutrition": nutrition_data
        })

    except Exception as e:
        print("Error performing recommendation:", e)
        return jsonify({"error": "Failed to perform recommendation", "details": str(e)}), 500

@app.route('/', methods=['GET'])
def hello():
    print("Hello endpoint accessed")
    return "Hello"

# Start the Flask server
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=3009, debug=True)
