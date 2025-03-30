from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Health check route
@app.route('/', methods=['GET'])
def home():
    return "✅ Receipt Extraction API is running now bhai!", 200

# Sample route to test file upload and workspace volume
@app.route('/process', methods=['POST'])
def process_receipt():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename

    # Save uploaded file to /workspace inside the container
    save_path = os.path.join('/workspace', filename)
    file.save(save_path)

    # Simulate processing and return success response
    return jsonify({
        "message": f"File '{filename}' received and saved to workspace.",
        "saved_path": save_path
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
