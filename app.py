from flask import Flask, request, redirect, url_for, render_template
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Pass the fs object to the run_demo function
            count, elapsed_time, file_id = demo.run_demo(filepath, fs)

            image_url = url_for('serve_pil_image', file_id=file_id)
            return render_template('result.html', count=count, elapsed_time=elapsed_time, image_url=image_url)
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
