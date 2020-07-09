from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import keras.backend as K
import librosa
import librosa.display

K.clear_session()

model = load_model('saved_models/weights.best.basic_cnn.hdf5')


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


# path = "free-spoken-digit-dataset/recordings/9_jackson_12.wav"

path = "spokenDigits/2untitled.wav"
class_label = "0"
data = extract_features(path)
features = [[data, class_label]]
#print(features)

featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

num_rows = 40
num_columns = 1
num_channels = 1

x = X.reshape(X.shape[0], num_rows, num_columns, num_channels)

# print(X.shape)
# X.reshape(1, num_rows, num_columns, num_channels)

result = model.predict(x)[0].tolist()
print(result)

print("predicted = ", result.index(max(result)))

