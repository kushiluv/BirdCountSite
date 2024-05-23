# utils.py
from io import BytesIO
from PIL import Image

def save_image_to_gridfs(fs, image, filename):
    img_io = BytesIO()
    image.save(img_io, format='PNG')
    img_io.seek(0)
    return fs.put(img_io, filename=filename, content_type='image/png')
