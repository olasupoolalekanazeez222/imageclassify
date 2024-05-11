
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from tkinter import *
from tkinter import messagebox
def runCode():
    first_input = train_name_entry.get()
    second_input = predict_name_entry.get()
    third_input = batch_name_entry.get()
    dataset_url = first_input #"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = pathlib.Path(data_dir).with_suffix('')

    batch_size = second_input
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)



    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()
    epochs=10
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    #data augmentation */
    data_augmentation = keras.Sequential(
    [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    ]
    )

    #Drop out */
    model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255), 
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()
    epochs = 15
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    sunflower_url = third_input #"https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    test_label = Label(root, text= "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    test_label.pack()
    messagebox.showinfo("Success", "Program is running !")



root = Tk()
root.title("Registration Form")
root.geometry('500x500')



# Create labels and entry fields for each input
first_name_label = Label(root, text="Zip folder should be uploaded as train data")
first_name_label.pack()

second_name_label = Label(root, text="Input the folder location url e.g GDrive/folder/yourzipfile")
second_name_label.pack()

third_name_label = Label(root, text=" Train data(80%) validation(20%) is extrated from the folder")
third_name_label.pack()

fourth_name_label = Label(root, text="Input the folder location url for predict data and batch size. e.g GDrive/folder/yourzipfile")
fourth_name_label.pack()

train_name_label = Label(root, text="Train Data")
train_name_label.pack()

train_name_entry = Entry(root)
train_name_entry.pack()

predict_name_label = Label(root, text="Predict Data")
predict_name_label.pack()

predict_name_entry = Entry(root)
predict_name_entry.pack()

batch_name_label = Label(root, text="Batch Size")
batch_name_label.pack()

batch_name_entry = Entry(root)
batch_name_entry.pack()

registerpredict_button = Button(root, text="Predict", command=runCode)
registerpredict_button.pack()

root.mainloop()




