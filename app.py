from flask import Flask, request, jsonify, render_template
import pipeline.full_pipeline as pipeline
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/extract', methods=['POST'])
def upload_receipt():
    if 'receipt' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['receipt']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Call your actual model logic here
    try:
        extracted_data = pipeline.exec(filepath)
        receipt_image = pipeline.ReceiptImage(filepath)
        image = pipeline.annotate_labels_onto_image(extracted_data, receipt_image)
        annotated_image_path = os.path.join("static/annotated", file.filename)
        image.save(annotated_image_path)

        return jsonify({
            'success': True,
            'filename': file.filename,
            'data': extracted_data,
            'annotated_image': annotated_image_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50505, debug=True)