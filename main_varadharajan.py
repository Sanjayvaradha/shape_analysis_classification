'''
Shape Analysis

Image folder - 360 defect images, 360 good images with random size

Resizing the images to 256x256 

Images loaded with the help of Keras data loader
Train, Validation, Test split - 440(62%),160(22%),120(16%)

Trained with 3 layers of CNN
Log metrics can be viewed in Tensorboard(Accuracy, Loss)
Training accuracy -> 98.6%
Validation accuracy -> 97.3%
Test accuracy -> 91.7%

Results and graphs are documented and attached along with this code.

'''
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

"""
Loading images of train and val in data_dir to Keras data loader
Resizing the image to 256x256 in keras loader(default) with batch size 32(default)
Rescaling the image to the range of 0 to 1 
"""

DATA_DIR = 'data'

data = tf.keras.utils.image_dataset_from_directory(DATA_DIR)
data = data.map(lambda x, y: (x/255, y))

"""Train and Validation split"""

train_size = int(len(data)*.7)+1
validation_size = int(len(data)*.3)

train = data.take(train_size)
validation = data.skip(train_size).take(validation_size)

"""Model to train the images with shape 256x256x3"""

EPOCHS = 40
LOSS = tf.losses.BinaryCrossentropy()
OPTIMIZER = 'adam'

model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=OPTIMIZER, loss=LOSS,
              metrics=['accuracy'])
LOGDIR = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)

hist = model.fit(train, epochs=EPOCHS, validation_data=validation,
                 callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

model.save("new_shape_analysis.h5")

"""
Loading the test images in test_dir to Keras data loader 
Resizing the image to 256x256(default) with batch size of 1 
Rescaling it to the range of 0 to 1
"""
model = load_model("new_shape_analysis.h5")
TEST_DIR = 'test'
test_data = tf.keras.utils.image_dataset_from_directory(TEST_DIR, batch_size=1)

test_data = test_data.map(lambda x, y: (x/255, y))

# Evaluation of test data
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

for batch in test_data.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

print(
    f'precision: {precision.result().numpy()}, recall: {recall.result().numpy()},accuracy: {accuracy.result().numpy()}')

# Displays test_images (4 images) along with true and predicted labels
fig, ax = plt.subplots(figsize=(6, 6))
for idx, batch in enumerate(test_data):
    if idx > 3:
        break
    ax = plt.subplot(2, 2, idx+1)
    X, y = batch
    pred = model.predict(X)
    actual = ["good" if i == 1 else "defect" for i in y]
    prediction = ["good" if i > 0.5 else "defect" for i in pred]

    plt.title(f'actual:{actual}\n,prediction:{prediction}')
    plt.imshow(X[0])
    ax.axis('off')
plt.show()

# Predicts the value for whole 120 images in test data

def predict(model):
    """
    Takes the argument trained model
    Returns the true class and predicted class of test data
    """
    predictions = []
    actual = []
    for batch in test_data.as_numpy_iterator():
        X, y = batch
        test_prediction = model.predict(X)
        predictions.append(test_prediction)
        actual.append(y)
    predictions_total = np.concatenate(predictions)
    actual_total = np.concatenate(actual)
    predictions_total = [1 if i > 0.5 else 0 for i in predictions_total]
    return actual_total, predictions_total


actual_total, predictions_total = predict(model)

# Plots the Confusion matrix and Classification report for the test data
confusionmatrix = confusion_matrix(
    actual_total, predictions_total, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusionmatrix, display_labels=['Defect', 'Good'])
disp.plot()
plt.show()


target_names = ['defect', 'good']
print(classification_report(actual_total,
      predictions_total, target_names=target_names))

# # To predict single image outside test_batch
# Uncomment and run the below script only if you are testing for a single image.


# def n_image(image):

#     img = cv2.imread(image)
#     resize = tf.image.resize(img, (256, 256))
#     yhat = model.predict(np.expand_dims(resize/255, 0))
#     prediction = ["good" if i > 0.5 else "defect" for i in yhat]

#     plt.imshow(resize.numpy().astype(int))
#     plt.title(f'prediction:{prediction}')
#     plt.show()


# """Give path to the image"""
# image = 'PATH TO THE IMAGE'
# n_image(image)
