import os
import json
import shutil
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
SYSTEM_INSTRUCTION = "You are a botanical AI expert specializing in identifying plant species from images.  \nYour task is to analyze the given plant image and provide the following details in a json format so i can load it in my app easily \n\n{\n  \"title\": \"Bamboo\",\n  \"description\": \"A fast-growing grass species known for its strength and versatility. Commonly used in construction, furniture, and paper production.\",\n  \"region\": \"Native to Asia, South America, and Africa; widely cultivated worldwide.\",\n  \"uses\": \"Used for construction, furniture, paper production, and as a food source (bamboo shoots).\",\n  \"scientific_name\": \"Bambusoideae\",\n  \"ecological_importance\": \"Helps prevent soil erosion, absorbs CO2, and provides habitat for wildlife.\",\n  \"medicinal_uses\": \"Bamboo leaves and shoots are used in traditional medicine for anti-inflammatory and digestive benefits.\"\n}\n\n\nIf the image is unclear or the plant cannot be identified with high confidence, state so explicitly and suggest possible species based on visible features.  \n"

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI app
app = FastAPI(title="Plant Identification API")

# Define model with instructions
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,  # Ensure response is concise
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=SYSTEM_INSTRUCTION,
)

def upload_to_gemini(path, mime_type="image/jpeg"):
    """Uploads image to Gemini API and returns file reference."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        return file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini upload failed: {str(e)}")

@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    """API endpoint to identify plant species from an uploaded image."""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload image to Gemini
        gemini_file = upload_to_gemini(temp_path)
        os.remove(temp_path)  # Cleanup temp file

        # Start chat session with Gemini model
        chat_session = model.start_chat(
            history=[{"role": "user", "parts": [gemini_file]}]
        )
        response = chat_session.send_message(
            "Identify this plant and provide details in JSON format with the following fields: "
            "'title', 'description', 'region', 'uses', 'scientific_name', 'ecological_importance', 'medicinal_uses'."
        )

        # Ensure response is structured JSON
        try:
            plant_info = json.loads(response.text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Gemini response is not valid JSON.")

        return {"plant_info": plant_info}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
