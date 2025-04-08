import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from imblearn.over_sampling import SMOTE


#add the csv data
df = pd.read_csv(r"C:\Users\DELL\Desktop\codes\project\heartdisease\heart_disease.csv")
df.describe()

target_col = 'Heart Disease Status'
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataset.")

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Handle missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

if num_cols:
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

if cat_cols:
    df[cat_cols] = df[cat_cols].fillna("Unknown")  # Avoid NoneType issues
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


X = df.drop(columns=[target_col])
y = df[target_col]

# Handle class imbalance using SMOTE
if len(set(y)) > 1:  # Ensure SMOTE only runs if multiple classes exist
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    print("Warning: SMOTE skipped due to only one class present in the dataset.")
    X_resampled, y_resampled = X, y

# Convert to DataFrame after SMOTE
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
if num_cols:
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols].copy())
    X_test[num_cols] = scaler.transform(X_test[num_cols].copy())



# Optimized Neural Network Model
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),  # Normalize activations for stable training
    Dropout(0.3),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks to improve training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("heart_disease_model.keras", save_best_only=True, mode='max')


# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr], verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Model Accuracy: {accuracy:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Classification report
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))


model.save("heart_disease_model.keras")