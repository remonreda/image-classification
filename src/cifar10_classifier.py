
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
NUM_CLASSES = 10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(smoke_test: bool = False):
    """Return ``(x_train, y_train), (x_test, y_test)`` with pixel values in [0, 255]."""
    if smoke_test:
        rng = np.random.default_rng(SEED)
        make = lambda n: (rng.integers(0, 256, (n, 32, 32, 3)).astype("float32"),
                          rng.integers(0, NUM_CLASSES, (n, 1)))
        return make(256), make(64)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train.astype("float32"), y_train), (x_test.astype("float32"), y_test)


def build_mlp() -> keras.Model:
    """Baseline: a single hidden layer on flattened pixels."""
    return keras.Sequential([
        keras.Input((32, 32, 3)),
        layers.Rescaling(1 / 255.0),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="mlp_baseline")


def build_cnn() -> keras.Model:
    """Small VGG-style CNN with data augmentation, batch-norm and dropout."""
    def block(filters):
        return [
            layers.Conv2D(filters, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(filters, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
        ]
    return keras.Sequential([
        keras.Input((32, 32, 3)),
        layers.Rescaling(1 / 255.0),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        *block(32), *block(64), *block(128),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="custom_cnn")


def build_resnet50(pretrained: bool = True) -> keras.Model:
    """Transfer learning: frozen ImageNet ResNet50 on 8x up-scaled images + a small head."""
    base = keras.applications.ResNet50(
        weights="imagenet" if pretrained else None,
        include_top=False, input_shape=(256, 256, 3))
    base.trainable = False
    return keras.Sequential([
        keras.Input((32, 32, 3)),
        layers.UpSampling2D(8),  
        layers.Lambda(keras.applications.resnet50.preprocess_input),
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="resnet50_transfer")


BUILDERS = {"mlp": build_mlp, "cnn": build_cnn, "resnet50": build_resnet50}
DEFAULT_EPOCHS = {"mlp": 20, "cnn": 30, "resnet50": 10}
LEARNING_RATES = {"mlp": 1e-3, "cnn": 1e-3, "resnet50": 1e-3}

def plot_history(history: keras.callbacks.History, name: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set_title(f"{name} - {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_history.png", dpi=130)
    plt.close(fig)


def train_model(key, data, args, out_dir: Path) -> dict:
    (x_train, y_train), (x_test, y_test) = data
    model = BUILDERS[key](pretrained=not args.smoke_test) if key == "resnet50" else BUILDERS[key]()
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATES[key]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=args.epochs or DEFAULT_EPOCHS[key],
        batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[RESULT] {model.name}: test accuracy = {acc:.4f}")
    plot_history(history, model.name, out_dir)
    model.save(out_dir / f"{model.name}.keras")
    return {"model": model.name, "test_loss": float(loss), "test_accuracy": float(acc),
            "epochs_run": len(history.history["loss"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=[*BUILDERS, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None, help="override per-model default epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--smoke-test", action="store_true",
                        help="run on tiny random data for 1 epoch (verifies the code path, no download)")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = args.epochs or 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)

    data = load_data(args.smoke_test)
    keys = list(BUILDERS) if args.model == "all" else [args.model]
    results = [train_model(k, data, args, args.output_dir) for k in keys]

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['model']:<20} test accuracy: {r['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
