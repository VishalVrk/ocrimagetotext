from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import requests
from io import BytesIO
from PIL import Image
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Gradio client
ocr_client = Client("vrkforever/OCR-image-to-text")
bot_client = Client("chuanli11/Chat-Llama-3.2-3B-Instruct-uncensored")

# API route for performing OCR
@app.route('/api/ocr', methods=['POST'])
def perform_ocr():
    data = request.get_json()
    image_url = data.get('imageUrl')
    
    if not image_url:
        return jsonify({"error": "Image URL is required."}), 400

    try:
        # Fetch the image from the provided URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL."}), 400

        # Convert the image to a file-like object for Gradio
        img_data = BytesIO(response.content)
        img = Image.open(img_data)

        # Save image temporarily in memory for Gradio
        img.save("temp_image.png")  # Save to a temp file for handle_file

        # Perform OCR using the Gradio client
      # Get the OCR result
        result = ocr_client.predict(
            Method="PaddleOCR",
            img=handle_file("temp_image.png"),  # Provide temp image to handle_file
            api_name="/predict"
        )

        # Prepare the input for bot_client by ensuring the OCR result is correctly inserted into the message
        bot_result = bot_client.predict(
            message=f"{result} \n based on the given value keep only ingredients. If no ingredients are found, say 'No Ingredients found'",
            system_prompt=f"{result} \n based on the given value keep only ingredients. If no ingredients are found, say 'No Ingredients found'",
            max_new_tokens=1024,
            temperature=0.6,
            api_name="/chat"
        )

        clean_output = bot_result.replace("assistant", "").strip()
        ingredients_list = clean_output.split("\n")
        ingredients_list = [ingredient.split(". ", 1)[1] if ". " in ingredient else ingredient for ingredient in ingredients_list]
        json_response = jsonify(ingredients_list)
       
        # Return the OCR result as a JSON response
        return json_response

    except Exception as e:
        return jsonify({"error": "Failed to perform OCR", "details": str(e)}), 500
    
@app.route('/api/recommend', methods=['POST'])
def perform_recomm():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract patient data and ingredients from the received data
    patient_data = data.get('patient')
    ingredients = data.get('ingredients')

    # Check if both patient data and ingredients exist
    if not patient_data or not ingredients:
        return jsonify({"error": "Missing patient data or ingredients"}), 400

    # Prepare the prompt for the recommendation
    prompt = f"""
    Patient Information:
    - Age: {patient_data.get('age')}
    - Allergies: {patient_data.get('allergies')}
    - Health Conditions: {patient_data.get('healthConditions')}
    - Sugar Level: {patient_data.get('sugarLevel')}
    - Blood Pressure: {patient_data.get('bloodPressure')}
    
    Ingredients:
    {ingredients}

    Based on this information, provide a recommendation if this food is safe for the patient to consume, and explain the reasons.
    """

    try:
        # Send the request to the bot client
        bot_result = bot_client.predict(
            message=prompt,
            system_prompt=prompt,
            max_new_tokens=1024,
            temperature=0.6,
            api_name="/chat"
        )

        # Clean the bot output
        clean_output = bot_result.replace("assistant", "").strip()

        # Return the recommendation to the frontend
        return jsonify({"recommendation": clean_output})

    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({"error": "Failed to perform recommendation", "details": str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(port=3009, debug=True)
