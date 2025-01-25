import os
import pandas as pd
import requests
import traceback
import csv
from flask import Flask, request, jsonify, send_file, send_from_directory
from io import BytesIO
import google.generativeai as genai
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the data
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SPECIES_DATA_DIR = os.path.join(BASE_DATA_DIR, 'MMAforays')
UPLOADS_DIR = os.path.join(BASE_DATA_DIR, 'uploads')

filename = os.path.join(UPLOADS_DIR, 'myspecies.csv')
data = pd.read_csv(filename)[['image_url', 'scientific_name', 'common_name']].dropna()
data = data.sample(frac=1).reset_index(drop=True)

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Pronunciation cache file
PRONUNCIATION_CACHE_FILE = 'pronounce.csv'

app.config['JSON_AS_ASCII'] = False

def save_pronunciation_cache(cache):
    """Save pronunciation cache to CSV in base data directory."""
    cache_file_path = os.path.join(BASE_DATA_DIR, 'pronounce.csv')
    with open(cache_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scientific_name', 'pronunciation'])
        for name, pronunciation in cache.items():
            writer.writerow([name, pronunciation])

def load_pronunciation_cache():
    """Load pronunciation cache from CSV in base data directory."""
    cache = {}
    cache_file_path = os.path.join(BASE_DATA_DIR, 'pronounce.csv')
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if exists
            for row in reader:
                if len(row) == 2:
                    cache[row[0]] = row[1]
    return cache

# Load pronunciation cache
pronunciation_cache = load_pronunciation_cache()

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
    
@app.route('/get_card', methods=['GET'])
def get_card():
    global data
    if data.empty:
        return jsonify({"message": "All flashcards completed!", "completed": True}), 200

    card = data.iloc[0].to_dict()
    return jsonify(card), 200


@app.route('/get_hints', methods=['GET'])
def get_hints():
    hints = data['scientific_name'].unique().tolist()
    return jsonify(hints), 200


@app.route('/check_answer', methods=['POST'])
def check_answer():
    payload = request.json
    user_answer = payload.get('answer', '').strip().lower()
    card = payload.get('card', {})

    if 'scientific_name' not in card:
        return jsonify({"correct": False, "message": "Invalid card data received."}), 400

    if user_answer == card['scientific_name'].lower():
        common_name = card.get('common_name', '')
        return jsonify({"correct": True, "message": f"Correct! ({common_name})"}), 200

    return jsonify({"correct": False, "message": "Incorrect. Try again!"}), 200


@app.route('/next_card', methods=['POST'])
def next_card():
    global data
    if not data.empty:
        data = data.iloc[1:].reset_index(drop=True)
    return jsonify({"message": "Next card loaded."}), 200


@app.route('/get_image', methods=['GET'])
def get_image():
    url = request.args.get('url')
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return send_file(BytesIO(response.content), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_csv_files', methods=['GET'])
def list_csv_files():
    directory = request.args.get('directory', 'MMAforays')
    directory_path = os.path.join(BASE_DATA_DIR, directory)
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    return jsonify({"files": csv_files}), 200


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    directory = request.form.get('directory', 'uploads')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(BASE_DATA_DIR, directory, filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully"}), 200

    return jsonify({"error": "Invalid file type"}), 400


@app.route('/select_csv', methods=['POST'])
def select_csv():
    payload = request.json
    filename = payload.get('filename')
    directory = payload.get('directory', 'MMAforays')

    file_path = os.path.join(BASE_DATA_DIR, directory, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        data = pd.read_csv(file_path)[['image_url', 'scientific_name', 'common_name']].dropna()
        data = data.sample(frac=1).reset_index(drop=True)
        return jsonify({
            "message": "CSV file selected successfully",
            "first_card": data.iloc[0].to_dict()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_csv/<filename>', methods=['DELETE'])
def delete_csv(filename):
    if not filename.endswith('.csv'):
        return jsonify({"error": "Invalid file type"}), 400

    file_path = os.path.join(os.getcwd(), filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(file_path)
        return jsonify({"message": "File deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/pronounce_name', methods=['POST'])
def pronounce_name():
    global pronunciation_cache
    payload = request.json
    scientific_name = payload.get('scientific_name', '')

    # Check cache first
    if scientific_name in pronunciation_cache:
        return jsonify({"pronunciation": pronunciation_cache[scientific_name]}), 200

    try:
        prompt = f"Pronounce {scientific_name} using English Scientific Latin with explanation"
        response = model.generate_content(prompt)
        pronunciation = response.text

        # Cache the pronunciation
        pronunciation_cache[scientific_name] = pronunciation

        # Save updated cache
        save_pronunciation_cache(pronunciation_cache)

        return jsonify({"pronunciation": pronunciation}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/exit_application', methods=['POST'])
def exit_application():
    try:
        # Write pronunciation cache
        if pronunciation_cache:
            save_pronunciation_cache(pronunciation_cache)

        # Trigger application exit
        os._exit(0)  # Forcefully exit the entire Python process
    except Exception as e:
        print(f"Exit application error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)