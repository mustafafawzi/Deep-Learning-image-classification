# vgg16_finetune.py
# ------------------------------------------------------------
# Finetuning van VGG16 conv base (na feature extraction) — multiclass
#
# Wat doet dit script?
# 1) Bouwt VGG16 + classifier head (zoals feature extraction)
# 2) Phase 1: traint head met conv base frozen
# 3) Phase 2: unfreezet enkel block5 van VGG16 + traint verder met lage LR
#
# Verwachte structuur (script staat IN deze map):
#   alle_schilders/
#     train/<schilder>/...
#     val/<schilder>/...
#     test/<schilder>/...
#
# Output:
#   alle_schilders/out_vgg16_finetune/
#     best_model_phase1.keras
#     best_model_phase2.keras
#     best_training_metrics_phase1.txt
#     best_training_metrics_phase2.txt
#     training_curves_phase1.png
#     training_curves_phase2.png
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

EPOCHS_PHASE1 = 30   # head training (met callbacks)
EPOCHS_PHASE2 = 20   # finetuning (met callbacks)

LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5     # lage LR voor finetuning

PATIENCE = 5

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
TEST_DIR  = BASE_DIR / "test"

OUT_DIR = BASE_DIR / "out_vgg16_finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_P1 = OUT_DIR / "best_model_phase1.keras"
BEST_MODEL_P2 = OUT_DIR / "best_model_phase2.keras"

tf.keras.utils.set_random_seed(SEED)



# Helpers
def save_training_curves(history, out_path: Path, title: str):
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

    plt.suptitle(title)
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

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("Confusion Matrix — VGG16 Finetuning (Testset)")
    fig.tight_layout()

    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print("Confusion matrix opgeslagen naar:", cm_path)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    print("Classification report opgeslagen naar:", report_path)

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


# Build model function
def build_vgg16_model(num_classes: int):
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Lambda(preprocess_input, name="vgg16_preprocess")(inputs)

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model



# Phase 1: Feature extraction (frozen base)
print("\n=== PHASE 1: Feature extraction (base frozen) ===")
model, base_model = build_vgg16_model(num_classes)

base_model.trainable = False  # freeze

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks_p1 = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath=str(BEST_MODEL_P1), monitor="val_loss", save_best_only=True),
]

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_p1
)

print("\nPhase 1 klaar. Beste model opgeslagen naar:", BEST_MODEL_P1)

write_best_training_metrics(history_p1, OUT_DIR / "best_training_metrics_phase1.txt")
save_training_curves(history_p1, OUT_DIR / "training_curves_phase1.png", title="Phase 1 — Frozen VGG16 (Feature Extraction)")



# Phase 2: Finetuning (unfreeze block5)
print("\n=== PHASE 2: Finetuning (unfreeze block5) ===")

# Start van de beste weights van phase 1
# (model heeft restore_best_weights=True, maar we laden ook expliciet de beste checkpoint)
model = keras.models.load_model(BEST_MODEL_P1)

# We moeten base_model terugvinden in het geladen model:
# - In deze architectuur is base_model een nested Model in de graph.
# - We vinden hem via naam "vgg16"
vgg16_in_loaded = None
for layer in model.layers:
    if isinstance(layer, keras.Model) and layer.name.startswith("vgg16"):
        vgg16_in_loaded = layer
        break
if vgg16_in_loaded is None:
    # fallback: zoek op class name
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "VGG16" in layer.__class__.__name__:
            vgg16_in_loaded = layer
            break

if vgg16_in_loaded is None:
    raise RuntimeError("Kon VGG16 base model niet terugvinden in het geladen model.")

# Unfreeze enkel block5
vgg16_in_loaded.trainable = True
set_trainable = False
for layer in vgg16_in_loaded.layers:
    if layer.name.startswith("block5"):
        set_trainable = True
    layer.trainable = set_trainable

# Recompile met lage learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_p2 = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath=str(BEST_MODEL_P2), monitor="val_loss", save_best_only=True),
]

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_p2
)

print("\nPhase 2 klaar. Beste model opgeslagen naar:", BEST_MODEL_P2)

write_best_training_metrics(history_p2, OUT_DIR / "best_training_metrics_phase2.txt")
save_training_curves(history_p2, OUT_DIR / "training_curves_phase2.png", title="Phase 2 — Finetuning VGG16 (block5 unfrozen)")



# Evaluate best finetuned model on test set
best_finetuned = keras.models.load_model(BEST_MODEL_P2)
evaluate_and_save(best_finetuned, test_ds, class_names, OUT_DIR)
