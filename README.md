# shape_analysis_classification
Analyzing the shape of a cell using NN

web.py script contains the HTML frontend interface

main.py script contains the FAST_API application

Run the main.py to deploy the trained model in local host.

Click the local host link, then do the following,

In URL, add "/docs" after 8001, the FAST-API application will open.

Click on the POST and "try it out" button.

Under the Request body section choose an image file(shapes from the test_images folder) to upload and click the "Execute" button predicts the image.

Under the Responses section, the prediction will be displayed with the class name either "good" or "bad" shape.

# DOCKER
The model has been dockerized and can be found on the below link:

https://hub.docker.com/r/sanjayggmu/shape_analysis_test
