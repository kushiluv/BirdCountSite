from flask import Flask, request, redirect, url_for, render_template, jsonify , json
import os
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from flask import make_response,abort
# Import your demo script
import demo
from pymongo import MongoClient
import gridfs

# MongoDB connection string
uri = "mongodb+srv://kushiluv:kushiluv25@cluster0.pety1ki.mongodb.net/"

# Connect to your MongoDB
client = MongoClient(uri)
db = client['BirdDatabase']
fs = gridfs.GridFS(db)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','CR2'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/save_annotations/<file_id>', methods=['POST'])
def save_annotations(file_id):
    try:
        data = request.json  # Get the annotations data from the request
        # Save the annotations data to MongoDB
        result = db.annotations.insert_one({
            'file_id': file_id,
            'annotations': data
        })
        print(result.inserted_id)
        
        return jsonify({'message': 'Annotations saved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/annotate/<file_id>')
def annotate_image(file_id):
    try:
        grid_out = fs.get(ObjectId(file_id))
        image_url = url_for('serve_pil_image', file_id=file_id)
        # Pass file_id to the template
        print("File ID:", file_id)
        return render_template('annotate.html', image_url=image_url, file_id=file_id)
    except gridfs.NoFile:
        abort(404)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save file to GridFS and get file_id
            file_id = fs.put(file.read(), filename=secure_filename(file.filename), content_type=file.content_type)
            
            # Process the file for prediction (adapted to accept a file_id)
            count, elapsed_time, heatmap_file_id = demo.run_demo(file_id, fs)  # Assuming run_demo is adapted to use file_id

            # Generate the URL for accessing the uploaded image
            image_url = url_for('serve_pil_image', file_id=str(heatmap_file_id))
            return render_template('result.html', count=count, elapsed_time=elapsed_time, image_url=image_url, file_id=str(file_id))
    return render_template('upload.html')
@app.route('/image/<file_id>')
def serve_pil_image(file_id):
    try:
        grid_out = fs.get(ObjectId(file_id))
        response = make_response(grid_out.read())
        response.mimetype = grid_out.content_type
        return response
    except gridfs.NoFile:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True)