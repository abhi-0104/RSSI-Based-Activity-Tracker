import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from scipy.signal import medfilt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

data = pd.read_csv('/content/filtered_data_new.csv')

reshaped_data = pd.DataFrame({
    'RSSI': pd.concat([data['Empty Room'], data['RUNNING'], data['SITTING'], data['Walking']]),
    'Activity': ['Empty Room'] * len(data) + ['Running'] * len(data) + ['Sitting'] * len(data) + ['Walking'] * len(data)
})

smote = SMOTE()
X, y = smote.fit_resample(reshaped_data[['RSSI']], reshaped_data['Activity'])

reshaped_data['RSSI'] = medfilt(reshaped_data['RSSI'], kernel_size=3)

imputer = SimpleImputer(strategy='median')
reshaped_data['RSSI'] = imputer.fit_transform(reshaped_data[['RSSI']])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(reshaped_data['Activity'])
y_encoded = to_categorical(y_encoded)

def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_sequences, y_sequences = create_sequences(X, y_encoded, time_steps)

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

model = Sequential()

# 1D Convolutional Layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=256, verbose=2)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'CNN + LSTM Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

