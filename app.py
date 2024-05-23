from flask import Flask, request, redirect, url_for, render_template, jsonify , json, session 
import os
from markupsafe import Markup
from base64 import b64decode
from flask import send_file
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from flask import make_response,abort
# Import your demo script
import model_files.demo as demo
import model_files.FSC_finetune_cross as FSC_finetune_cross
from pymongo import MongoClient
import gridfs
from io import BytesIO
from PIL import Image
from flask_session import Session
import matplotlib
matplotlib.use('Agg')
import pymongo
import json
import urllib.request
import os
from urllib.error import HTTPError, URLError
import json
import math
from PIL import Image
import json
from random import shuffle
import warnings  
import shutil
warnings.filterwarnings('ignore')

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
from flask import jsonify
import os
import json



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Handle the file upload functionality.

    This function is responsible for handling the file upload functionality of the web application.
    It checks if a file is uploaded, saves the file to GridFS, processes the file for prediction,
    and renders the result template with the predicted count, elapsed time, and image URLs.

    Returns:
        If the request method is 'POST' and the file is successfully uploaded and processed,
        it renders the 'result.html' template with the predicted count, elapsed time, and image URLs.
        If the request method is 'GET' or the file upload fails, it renders the 'upload.html' template.
    """
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/get_checkpoint_data')
def get_checkpoint_data():
    """
    Retrieves the checkpoint data and prepares it to send to the frontend.

    Returns:
        A JSON response containing the checkpoint data.
    """
    # Path to the directory where checkpoint files are stored
    checkpoint_directory = "model_files/pth"
    checkpoint_files = os.listdir(checkpoint_directory)
    print(checkpoint_files)
    # Path to the parameters.json file
    parameters_path = "./parameters.json"
    
    # Load the MAE and other data
    if os.path.exists(parameters_path):
        with open(parameters_path, 'r') as file:
            parameters = json.load(file)
    else:
        parameters = []

    # Prepare data to send to the frontend
    data = {
        "checkpoints": [
            {"name": os.path.splitext(f)[0], "path": os.path.join(checkpoint_directory, f), "MAE": next((p['MAE'] for p in parameters if p['path_file'] == f), None)}
            for f in checkpoint_files if f.endswith(".pth")
        ]
    }
    print(data)
    return jsonify(data)

@app.route('/run_finetune_script')
def run_finetune_script():  
    """
    This function is responsible for running the finetune script.
    It performs the following steps:
    1. Clears the finetune_data_mongo folder.
    2. Fetches approved entries from the BirdDatabase collection.
    3. Downloads images and saves annotation data as JSON.
    4. Renames image files and reads JSON files.
    5. Merges annotations into a single JSON file.
    6. Splits the images into test, validation, and training sets.
    7. Saves the split data to a new JSON file.
    8. Executes the FSC_finetune_cross.run_finetune() function.
    9. Returns a JSON response indicating the successful execution of the script.
    """
    # Clear the finetune_data_mongo folder
    shutil.rmtree("model_files/finetune_data_mongo")
    os.makedirs("model_files/finetune_data_mongo")
    # print("insideeeeeee")
    # print("Running the script...")
    client = pymongo.MongoClient(uri)
    db = client["BirdDatabase"]
    collection = db["annotations"]

    # Storage directory setup
    
    storage_dir = "model_files/finetune_data_mongo/Images"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Fetching approved entries
    entries = collection.find({"approved": True})

    for entry in entries:
        file_id = entry["file_id"]
        annotations = entry.get("annotations", [])
        bbox = entry.get("bbox", [])
        # Ensuring the filename is unique by appending the ObjectId
        object_id = str(entry["_id"])
        image_url = f"http://127.0.0.1:8080/image/{file_id}"
        image_filename = os.path.join(storage_dir, f"{file_id}_{object_id}.png")

        try:
            # Downloading the image
            urllib.request.urlretrieve(image_url, image_filename)

            # Saving annotation data as JSON
            json_filename = os.path.join(storage_dir, f"{file_id}_{object_id}.json")
            with open(json_filename, 'w') as json_file:
                data_to_save = {
                    "annotations": annotations,
                    "bbox": bbox
                }
                json.dump(data_to_save, json_file, indent=4)

        except HTTPError as e:
            print(f"HTTP Error downloading image for file ID {file_id}: {e}")
        except URLError as e:
            print(f"URL error encountered for file ID {file_id}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("All approved entries have been processed, files saved for available images only.")



    directory = "model_files/finetune_data_mongo/Images"
    merged_annotations = {}

    # Rename image files and read JSON files
    for i, filename in enumerate(sorted(os.listdir(directory)), start=1):
        if filename.endswith(".png"):
            old_image_path = os.path.join(directory, filename)
            new_image_name = f"{i//2}.png"  # Renaming logic may need adjustment to match your requirements
            new_image_path = os.path.join(directory, new_image_name)
            os.rename(old_image_path, new_image_path)

            # Open the image to fetch its dimensions
            with Image.open(new_image_path) as img:
                width, height = img.size
            
            # Assuming JSON file has a similar name with .json extension
            old_json_path = os.path.join(directory, filename.replace('.png', '.json'))

            if os.path.exists(old_json_path):
                with open(old_json_path, 'r') as json_file:
                    data = json.load(json_file)

                points = []
                box_examples_coordinates = []

                # Process point annotations
                if "annotations" in data:
                    for item in data["annotations"]:
                        attrs = json.loads(item["region_shape_attributes"])
                        points.append([attrs["cx"], attrs["cy"]])

                # Process bounding box annotations
                if "bbox" in data:
                    for bbox in data["bbox"]:
                        box = []
                        for coord in bbox:
                            # Apply math.floor to each coordinate
                            box.append([math.floor(coord[0]), math.floor(coord[1])])
                        box_examples_coordinates.append(box)

                # Prepare the structured data for this image
                merged_annotations[new_image_name] = {
                    "H": height,
                    "W": width,
                    "points": points,
                    "box_examples_coordinates": box_examples_coordinates
                }
                os.remove(old_json_path)  # Optionally remove the old JSON file

    # Save the merged annotations into a single JSON file
    merged_json_path = os.path.join("model_files/finetune_data_mongo", "merged_annotations.json")
    with open(merged_json_path, 'w') as outfile:
        json.dump(merged_annotations, outfile, indent=4)

    print("Images and JSON files have been processed.")


    directory = "model_files/finetune_data_mongo"
    merged_json_path = os.path.join(directory, "merged_annotations.json")

    # Load merged annotations
    with open(merged_json_path, 'r') as file:
        data = json.load(file)

    # Extract all image names
    image_files = list(data.keys())
    shuffle(image_files)  # Shuffle to randomize distribution

    # Define split proportions
    num_images = len(image_files)
    test_size = int(0.1 * num_images)
    test_coco_size = int(0.1 * num_images)
    val_size = int(0.1 * num_images)
    val_coco_size = int(0.1 * num_images)
    train_size = num_images - (test_size + test_coco_size + val_size + val_coco_size)

    # Split the images based on calculated sizes
    splits = {
        "test": image_files[:test_size],
        "test_coco": image_files[test_size:test_size + test_coco_size],
        "val": image_files[test_size + test_coco_size:test_size + test_coco_size + val_size],
        "val_coco": image_files[test_size + test_coco_size + val_size:test_size + test_coco_size + val_size + val_coco_size],
        "train": image_files[test_size + test_coco_size + val_size + val_coco_size:]
    }

    # Save the split data to a new JSON file
    split_json_path = os.path.join(directory, "data_split.json")
    with open(split_json_path, 'w') as outfile:
        json.dump(splits, outfile, indent=4)

    print("Data has been split and saved.")
    FSC_finetune_cross.run_finetune()

    return jsonify({'message': 'The script has been executed successfully!'})

@app.route('/annotate/<file_id>')
def annotate_image(file_id):
    """
    Annotates an image with cluster centers.

    Args:
        file_id (str): The ID of the file to be annotated.

    Returns:
        A rendered HTML template with the annotated image and other relevant information.
    """
    try:
        grid_out = fs.get(ObjectId(file_id))
        image_url = url_for('serve_pil_image', file_id=file_id)
        
        # Retrieve cluster_centers from session
        cluster_centers = session.get('cluster_centers', [])
        print(cluster_centers)
        return render_template('annotate.html', image_url=image_url, file_id=str(file_id), cluster_centers=cluster_centers, image_dimensions=session['image_dimensions'])
    except gridfs.NoFile:
        abort(404)

@app.route('/save_annotations/<file_id>', methods=['POST'])
def save_annotations(file_id):
    """
    Save annotations and bounding box data to MongoDB.

    This function receives a POST request containing annotations and bounding box data for a specific file.
    The annotations and bounding box data are extracted from the request and converted into the appropriate format.
    The image file associated with the annotations is saved to GridFS, and the annotations, bounding box data, and image ID are stored in MongoDB.

    Parameters:
    - file_id (str): The ID of the file being annotated.

    Returns:
    - JSON response: A JSON response indicating the success or failure of the operation. If successful, the response includes a message and the image ID.
    """

    try:
        # Extract annotations and bounding box data from the request
        annotations = json.loads(request.form.get('annotations'))
        bbox_data = json.loads(request.form.get('boxes'))
        
        # Convert bbox data to corner coordinates
        box_examples_coordinates = []
        for bbox in bbox_data:
            # Extract coordinates and handle negative width or height
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']

            if width < 0:
                x += width  # Move x to the left
                width = -width  # Make width positive
            if height < 0:
                y += height  # Move y up
                height = -height  # Make height positive

            # Calculate corners
            top_left = [x, y]
            top_right = [x + width, y]
            bottom_left = [x, y + height]
            bottom_right = [x + width, y + height]
            
            # Append as a list of four coordinates
            box_examples_coordinates.append([top_left, top_right, bottom_right, bottom_left])

        # Get the image file from the form
        image = request.files.get('image')
        if image:
            # Save the image file to GridFS
            image_id = fs.put(image.read(), content_type=image.content_type)
        else:
            image_id = None

        # Save the annotations and bbox to MongoDB
        result = db.annotations.insert_one({
            'file_id': file_id,
            'image_id': image_id,  # Save the GridFS image ID here
            'annotations': annotations,
            'bbox': box_examples_coordinates,
            'approved': False
        })

        return jsonify({'message': 'Annotations saved successfully', 'image_id': str(image_id)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



@app.route('/about')
def about_file():
    """
    Renders the about.html template.

    Returns:
        The rendered about.html template.
    """
    return render_template('about.html')

@app.route('/contact')
def contact_file():
    """
    Renders the contact.html template.

    Returns:
        The rendered contact.html template.
    """
    return render_template('contact.html')

@app.route('/load_model', methods=['POST'])
def load_new_model():
    """
    Load a new model based on the provided checkpoint path.

    Args:
        None

    Returns:
        A JSON response indicating the success or failure of the model loading process.

    Raises:
        None
    """
    checkpoint_path = request.json['checkpoint']
    print(checkpoint_path)
    global model, model_without_ddp
    model = demo.load_model(checkpoint_path)  # Load the new model
    model_without_ddp = model  # Update the global variable if necessary
    return jsonify({'message': 'Model loaded successfully'})



@app.route('/adjust_clusters', methods=['POST'])
def adjust_clusters():
    """
    Adjusts the clusters based on the slider value.

    Returns:
        If the adjustment is successful, returns the selected clusters as JSON.
        If the cluster data is not available in the session, returns an error message as JSON with status code 404.
        If the slider value is out of range, returns an error message as JSON with status code 400.
        If an exception occurs during the adjustment process, returns an error message as JSON with status code 500.
    """
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
    """
    Retrieve image information from GridFS based on the provided file ID.

    Args:
        file_id (str): The ID of the file in GridFS.

    Returns:
        tuple: A tuple containing the JSON response and the HTTP status code.
            The JSON response contains the file ID, size, and mode of the image.
            The HTTP status code indicates the success or failure of the request.
    """
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
    """
    Serve a PIL image from MongoDB GridFS based on the given file ID.

    Args:
        file_id (str): The ID of the file in MongoDB GridFS.

    Returns:
        flask.Response: The response object containing the image data.

    Raises:
        404: If the file with the given ID does not exist in GridFS.
    """
    try:
        grid_out = fs.get(ObjectId(file_id))
        response = make_response(grid_out.read())
        response.mimetype = grid_out.content_type
        return response
    except gridfs.NoFile:
        abort(404)
        
@app.route('/admin/review_annotations')
def review_annotations():
    """
    Fetches the annotations that have not been approved yet and renders them on the review_annotations.html template.

    Returns:
        The rendered template with the annotations data.
    """
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



@app.route('/admin/approve_annotation/<annotation_id>')
def approve_annotation(annotation_id):
    """
    Approves an annotation with the given ID.

    Parameters:
    - annotation_id (str): The ID of the annotation to be approved.

    Returns:
    - JSON response: A JSON response indicating the result of the approval process.
        - If the annotation is approved successfully, the response will have a status code of 200 and a message indicating success.
        - If an error occurs during the approval process, the response will have a status code of 500 and an error message.

    """
    try:
        db.annotations.update_one({'_id': ObjectId(annotation_id)}, {'$set': {'approved': True}})
        return jsonify({'message': 'Annotation approved successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/reject_annotation/<annotation_id>')
def reject_annotation(annotation_id):
    """
    Deletes an annotation from the database and redirects to the review_annotations page.

    Parameters:
    - annotation_id (str): The ID of the annotation to be deleted.

    Returns:
    - redirect: Redirects to the review_annotations page.
    """
    db.annotations.delete_one({'_id': ObjectId(annotation_id)})
    return redirect(url_for('review_annotations'))


@app.route('/annotated_image/<image_id>')
def serve_annotated_image(image_id):
    """
    Serve the annotated image with the given image_id.

    Args:
        image_id (str): The ID of the image to serve.

    Returns:
        flask.Response: The response containing the annotated image.

    Raises:
        404: If the image with the given ID does not exist.
    """
    try:
        grid_out = fs.get(ObjectId(image_id))
        response = make_response(grid_out.read())
        response.mimetype = grid_out.content_type
        return response
    except gridfs.NoFile:
        abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)