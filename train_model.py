# train_model.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
#  Dataset paths
# ------------------------------
train_dir = 'skin dataset/dataset/train'
val_dir = 'skin dataset/dataset/val'

# ------------------------------
#  Data generators with augmentation
# ------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    brightness_range=[0.7, 1.3]
)

val_gen = ImageDataGenerator(rescale=1./255)  # validation should be simple

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ------------------------------
#  MobileNetV2 Transfer Learning
# ------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------------------
#  Early stopping callback
# ------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ------------------------------
#  Train the model
# ------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,  # you can increase if needed
    callbacks=[early_stop]
)

# ------------------------------
#  Save the trained model
# ------------------------------
model.save('skin_model.h5')
print(" Training complete. Model saved as 'skin_model.h5'")

# ------------------------------
#  Optional: Save class labels
# ------------------------------
import pickle
with open('class_labels.pkl', 'wb') as f:
    pickle.dump(train_data.class_indices, f)
print(" Class labels saved as 'class_labels.pkl'")
