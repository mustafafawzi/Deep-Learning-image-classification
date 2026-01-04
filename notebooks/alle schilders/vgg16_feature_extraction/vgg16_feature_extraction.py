# vgg16_feature_extraction.py
# ------------------------------------------------------------
# Feature extraction met VGG16 (conv base frozen) — multiclass
# Te runnen op de Vlaamse Supercomputer (VSC).
#
# Verwachte structuur (script staat IN deze map):
#   alle_schilders/
#     train/<schilder>/...
#     val/<schilder>/...
#     test/<schilder>/...
#
# Output:
#   alle_schilders/out_vgg16_feature_extraction/
#     best_model.keras
#     best_training_metrics.txt
#     training_curves.png
#     test_metrics.txt
#     confusion_matrix.png
#     classification_report.txt
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 30

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
TEST_DIR  = BASE_DIR / "test"

OUT_DIR = BASE_DIR / "out_vgg16_feature_extraction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_model.keras"

tf.keras.utils.set_random_seed(SEED)


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


def write_best_training_metrics(history, out_path: Path):
    hist = history.history
    best_val_loss_epoch = int(np.argmin(hist["val_loss"]))
    best_val_acc_epoch  = int(np.argmax(hist["val_accuracy"]))

    lines = [
        "=== BEST TRAIN/VAL METRICS SUMMARY ===",
        "",
        f"Best val_loss epoch: {best_val_loss_epoch + 1}",
        f"  train_loss: {float(hist['loss'][best_val_loss_epoch]):.4f}",
        f"  val_loss  : {float(hist['val_loss'][best_val_loss_epoch]):.4f}",
        f"  train_acc : {float(hist['accuracy'][best_val_loss_epoch]):.4f}",
        f"  val_acc   : {float(hist['val_accuracy'][best_val_loss_epoch]):.4f}",
        "",
        f"Best val_accuracy epoch: {best_val_acc_epoch + 1}",
        f"  train_loss: {float(hist['loss'][best_val_acc_epoch]):.4f}",
        f"  val_loss  : {float(hist['val_loss'][best_val_acc_epoch]):.4f}",
        f"  train_acc : {float(hist['accuracy'][best_val_acc_epoch]):.4f}",
        f"  val_acc   : {float(hist['val_accuracy'][best_val_acc_epoch]):.4f}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("Best train/val metrics opgeslagen naar:", out_path)


def evaluate_and_save(model, test_ds, class_names, out_dir: Path):
    test_loss, test_acc = model.evaluate(test_ds)
    print("\nTest loss:", test_loss)
    print("Test accuracy:", test_acc)

    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.concatenate([y for _, y in test_ds]).astype(int).ravel()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("Confusion Matrix — VGG16 Feature Extraction (Testset)")
    fig.tight_layout()

    cm_path = out_dir / "confusion_matrix.png"
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
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    print("Classification report opgeslagen naar:", report_path)

    # Save test metrics
    metrics_path = out_dir / "test_metrics.txt"
    metrics_path.write_text(f"test_loss={test_loss}\ntest_accuracy={test_acc}\n")
    print("Test metrics opgeslagen naar:", metrics_path)


# Load datasets
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
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n--- Test Data ---")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nKlassen: {class_names}")
print(f"Num classes: {num_classes}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)


# Model: VGG16 conv base frozen (feature extraction)
inputs = keras.Input(shape=IMG_SIZE + (3,))

# VGG16 preprocessing (NIET Rescaling 1./255)
x = layers.Lambda(preprocess_input, name="vgg16_preprocess")(inputs)

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # <-- FEATURE EXTRACTION

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Callbacks
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

# Save best train/val metrics + curves
best_metrics_path = OUT_DIR / "best_training_metrics.txt"
write_best_training_metrics(history, best_metrics_path)

curves_path = OUT_DIR / "training_curves.png"
save_training_curves(history, curves_path)
print("Training curves opgeslagen naar:", curves_path)

# Test evaluation + artifacts
evaluate_and_save(model, test_ds, class_names, OUT_DIR)
