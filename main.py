import pandas as pd
import librosa
import librosa.display
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from datetime import datetime


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


path = "free-spoken-digit-dataset/recordings/"

files = [f for f in listdir(path) if isfile(join(path, f))]
features = []
# Iterate through each sound file and extract the features
for file_name in files:
    class_label = file_name[:1]
    data = extract_features(path + file_name)
    features.append([data, class_label])


# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ' + str(len(featuresdf)) + ' files')

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

#print("size x=", len(X))

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
print("\nx_test=", x_test, "\nX=", X)
# print(x_train, "\n", len(x_train), "\n xtrain.shape=", x_train.shape)

num_rows = 40
num_columns = 1
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]

filter_size = 2

# Construct model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding="same", input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(padding="same", pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(padding="same", filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(padding="same", pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(padding="same", filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(padding="same", pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(padding="same", filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(padding="same", pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 72*6
num_batch_size = 256*6

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
