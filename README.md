# Bird Count

## Description

This repository contains the code for a web application that estimates bird counts from uploaded images and allows users to help improve the model through annotations.

## How to Run the App

To run the app, perform the following steps:

1. Clone the repository.
2. Download the model's weights from [this link](https://drive.google.com/file/d/1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ/view) and save it in `model_files/pth`.
3. Rename the `.pth` file as `original.pth`.
4. Download the DB from [this link](https://drive.google.com/file/d/14BAPW6Ah-d-8Gr0bFE0DZ_6QgW-GSP1t/view?usp=sharing) and save it in `model_files/eval_db`.
5. Ensure you're using Python v3.9 and install the dependencies by running the command: `pip install -r requirements.txt`.
6. Run the site using the command: `python app.py`.

## How to Navigate the Site

1. On the home page, use the "Upload Image" button to upload an image for bird count estimation.
![image](https://github.com/kushiluv/BirdCountSite/assets/88649199/faa4d919-7e1a-468d-8667-c73b4b7e53a1)
2. If you are satisfied with the result, you can try uploading more images.
![image](https://github.com/kushiluv/BirdCountSite/assets/88649199/57248e41-4cd5-4196-8d28-470ad775d07c)
3. If you are not satisfied, you can choose the "Help Us Improve" button to make annotations for us.
4. In the annotation tab, use the slider to get the optimal annotation.
![image](https://github.com/kushiluv/BirdCountSite/assets/88649199/43d11b0a-6914-4b06-b809-93941ec853f6)
5. Click on an empty surface to annotate or click on an existing annotation to disable that dot.
6. After this step, draw 3-4 bounding boxes using the cursor around singular birds so that the model can analyze what it is counting.
7. Click on the "Submit Annotation" button.
8. After multiple annotations, you can use the URL `/admin/review_annotations` to access the admin portal and review annotations.
![image](https://github.com/kushiluv/BirdCountSite/assets/88649199/657a9101-5c4a-44b5-95ce-498f8d92c3e9)
9. Approve or deny the annotations as necessary.
10. Use the "Fine-tune" button to fine-tune the model. The fine-tuned `.pth` file will be stored on the site.
11. Using the drop-down menu, select the model with the least Mean Absolute Error (MAE) to be used in annotation from this point forward.
