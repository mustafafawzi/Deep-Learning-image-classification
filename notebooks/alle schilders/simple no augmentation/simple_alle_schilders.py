# train_all_schilders_simple.py  (met callbacks toegevoegd)
# ------------------------------------------------------------
# Simpele ConvNet training (alle schilders) volgens methode:
# - geen augmentation
# - geen class weights
# - cache().shuffle().prefetch()
# + callbacks: EarlyStopping + ModelCheckpoint (best practice)
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# 
# Configuratie
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
SEED = 123
EPOCHS = 50  # mag hoger: callbacks stoppen automatisch

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
TEST_DIR  = BASE_DIR / "test"

OUT_DIR = BASE_DIR / "out_simple_convnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_model.keras"


# Helpers
def save_training_curves(history, out_path: Path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("accuracy", []), label="train acc")
    plt.plot(epochs, hist.get("val_accuracy", []), label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("loss", []), label="train loss")
    plt.plot(epochs, hist.get("val_loss", []), label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# Datasets
print("--- Training Data ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\n--- Validation Data ---")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\n--- Test Data ---")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nKlassen: {class_names}")
print(f"Num classes: {num_classes}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Model (simpel, geen augmentation)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.summary()


# Callbacks (toegevoegd)
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),
]


# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining klaar. Beste model opgeslagen naar:", BEST_MODEL_PATH)

# Save curves + history
curves_path = OUT_DIR / "training_curves.png"
save_training_curves(history, curves_path)
print("Training curves opgeslagen naar:", curves_path)

history_csv = OUT_DIR / "history.csv"
pd.DataFrame(history.history).to_csv(history_csv, index=False)
print("History CSV opgeslagen naar:", history_csv)


# Test + confusion matrix + report
test_loss, test_acc = model.evaluate(test_ds)
print("\nTest loss:", test_loss)
print("Test accuracy:", test_acc)

# Predict
y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)

# True labels
y_true = np.concatenate([y for _, y in test_ds]).astype(int).ravel()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
ax.set_title("Confusion Matrix — Alle Schilders (Testset)")
fig.tight_layout()

cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=200)
plt.close(fig)
print("Confusion matrix opgeslagen naar:", cm_path)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=3
)
report_path = OUT_DIR / "classification_report.txt"
report_path.write_text(report)
print("Classification report opgeslagen naar:", report_path)

# Save test metrics
metrics_path = OUT_DIR / "test_metrics.txt"
metrics_path.write_text(f"test_loss={test_loss}\ntest_accuracy={test_acc}\n")
print("Test metrics opgeslagen naar:", metrics_path)
