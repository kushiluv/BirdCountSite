from flask import Flask, request, redirect, url_for, render_template, jsonify , json, session 
import os
from markupsafe import Markup
from base64 import b64decode
from flask import send_file
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from flask import make_response,abort
# Import your demo script
import demo
from pymongo import MongoClient
import gridfs
from io import BytesIO
from PIL import Image
from flask_session import Session
# MongoDB connection string
uri = "mongodb+srv://kushiluv:kushiluv25@cluster0.pety1ki.mongodb.net/"

# Connect to your MongoDB
client = MongoClient(uri)
db = client['BirdDatabase']
fs = gridfs.GridFS(db)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_herae' 
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 16MB limit
app.config["SESSION_TYPE"] = "filesystem"  # Or another server-side storage
Session(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','CR2'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/save_annotations/<file_id>', methods=['POST'])
def save_annotations(file_id):
    try:
        # Get the annotations data from the form
        annotations = json.loads(request.form.get('annotations'))
        file_id = request.form.get('file_id')
        
        # Get the image file from the form
        image = request.files.get('image')
        if image:
            # Save the image file to GridFS
            image_id = fs.put(image.read(), content_type=image.content_type)
        else:
            image_id = None

        # Save the annotations data to MongoDB
        result = db.annotations.insert_one({
            'file_id': file_id,
            'image_id': image_id,  # Save the GridFS image ID here
            'annotations': annotations
        })

        return jsonify({'message': 'Annotations saved successfully', 'image_id': str(image_id)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/annotate/<file_id>')
def annotate_image(file_id):
    try:
        grid_out = fs.get(ObjectId(file_id))
        image_url = url_for('serve_pil_image', file_id=file_id)
        
        # Retrieve cluster_centers from session
        cluster_centers = session.get('cluster_centers', [])
        print(cluster_centers)
        return render_template('annotate.html', image_url=image_url, file_id=str(file_id), cluster_centers=cluster_centers, image_dimensions=session['image_dimensions'])
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
           
            count, elapsed_time, heatmap_file_id, cluster_centers, image_dimensions = demo.run_demo(file_id, fs)
            session['image_dimensions'] = image_dimensions
            global image_url ; image_url= url_for('serve_pil_image', file_id=str(file_id))
            global density_map_url ; density_map_url = url_for('serve_pil_image', file_id=str(heatmap_file_id))
            session['pre_computed_clusters'] = cluster_centers
            session.modified = True 
            print(session.get('pre_computed_clusters')[0])
        # Pass cluster_centers to the template
            
            # Generate the URL for accessing the uploaded image
            
            return render_template('result.html', count=count, elapsed_time=elapsed_time, image_url=image_url, file_id=str(file_id), cluster_centers=cluster_centers[0], density_map_url=density_map_url)
    return render_template('upload.html')

@app.route('/adjust_clusters', methods=['POST'])
def adjust_clusters():
    print(session.get('image_dimensions'))
    print(session.get('pre_computed_clusters')) 
    if 'pre_computed_clusters' not in session or not session['pre_computed_clusters']:
        return jsonify({'error': 'Cluster data not available'}), 404
    try:
        slider_value = request.json.get('slider_value', 0)
        slider_value = int(slider_value)
        cluster_sets = session.get('pre_computed_clusters', [])
        print(f"Slider value: {slider_value}, Number of cluster sets: {len(cluster_sets)}")
        
        if slider_value < len(cluster_sets):
            selected_clusters = cluster_sets[slider_value]
            return jsonify(selected_clusters)
        else:
            return jsonify({'error': 'Slider value out of range'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/image_info/<file_id>')

def image_info(file_id):
    try:
        # Retrieve the file from GridFS
        grid_out = fs.get(ObjectId(file_id))
        # Create a BytesIO object from the file data
        image_data = BytesIO(grid_out.read())
        # Load the image using PIL
        image = Image.open(image_data)
        # Get image size and mode
        size = image.size
        mode = image.mode
        # Return the information as JSON
        return jsonify({'file_id': file_id, 'size': size, 'mode': mode}), 200
    except gridfs.NoFile:
        return jsonify({'error': 'File not found'}), 404    
@app.route('/image/<file_id>')
def serve_pil_image(file_id):
    try:
        grid_out = fs.get(ObjectId(file_id))
        response = make_response(grid_out.read())
        response.mimetype = grid_out.content_type
        return response
    except gridfs.NoFile:
        abort(404)
@app.route('/admin/review_annotations')
def review_annotations():
    # Fetch only the annotations that have not been approved yet
    annotations_list = db.annotations.find({'approved': {'$ne': True}})
    annotations = []
    for annotation in annotations_list:
        annotations.append({
            '_id': str(annotation['_id']),
            'file_id': annotation['file_id'],
            'image_id': str(annotation.get('image_id', '')),
            'annotations': annotation['annotations']
        })
    annotations_data = {'annotations': annotations}
    return render_template('review_annotations.html', annotations_data=annotations_data)



    return render_template('review_annotations.html', annotations_data=annotations_data)

@app.route('/admin/approve_annotation/<annotation_id>')
def approve_annotation(annotation_id):
    try:
        db.annotations.update_one({'_id': ObjectId(annotation_id)}, {'$set': {'approved': True}})
        return jsonify({'message': 'Annotation approved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/reject_annotation/<annotation_id>')
def reject_annotation(annotation_id):
    db.annotations.delete_one({'_id': ObjectId(annotation_id)})
    return redirect(url_for('review_annotations'))
@app.route('/annotated_image/<image_id>')
def serve_annotated_image(image_id):
    try:
        grid_out = fs.get(ObjectId(image_id))
        response = make_response(grid_out.read())
        response.mimetype = grid_out.content_type
        return response
    except gridfs.NoFile:
        abort(404)
if __name__ == '__main__':
    app.run(debug=True)