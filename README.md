# shape_analysis_classification
Analyzing the shape of a cell using NN

In main.py script contains the FAST_API application

Run the main.py to deploy the trained model in local host.

Click the local host link, if details not found page appears, then do the following,

In URL, add "/docs" after 8000, the FAST-API application will open.

Click on POST and "try it out" button.

Under Resquest body section choose an image file(shapes from test_images folder) to upload and click "Execute" button predict the image.

Under Responses section the prediction will be displayed with class name either "good" or "bad" shape.
