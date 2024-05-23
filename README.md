# Bird Count

## Description

This repository contains the code for a web application that estimates bird counts from uploaded images and allows users to help improve the model through annotations.

## How to Run the App

To run the app, perform the following steps:

1. Clone the repository.
2. Download the model's weights from [this link](https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view) and save it in `model_files/pth`.
3. Rename the `.pth` file as `original.pth`.
4. Download the DB from [this link](https://example.com/db) and save it in `model_files/eval_db`.
5. Ensure you're using Python v3.9 and install the dependencies by running the command: `pip install -r requirements.txt`.
6. Run the site using the command: `python app.py`.

## How to Navigate the Site

1. On the home page, use the "Upload Image" button to upload an image for bird count estimation.
2. If you are satisfied with the result, you can try uploading more images.
3. If you are not satisfied, you can choose the "Help Us Improve" button to make annotations for us.
4. In the annotation tab, use the slider to get the optimal annotation.
5. Click on an empty surface to annotate or click on an existing annotation to disable that dot.
6. After this step, draw 3-4 bounding boxes using the cursor around singular birds so that the model can analyze what it is counting.
7. Click on the "Submit Annotation" button.
8. After multiple annotations, you can use the URL `/admin/review_annotations` to access the admin portal and review annotations.
9. Approve or deny the annotations as necessary.
10. Use the "Fine-tune" button to fine-tune the model. The fine-tuned `.pth` file will be stored on the site.
11. Using the drop-down menu, select the model with the least Mean Absolute Error (MAE) to be used in annotation from this point forward.
