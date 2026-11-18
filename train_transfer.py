"""Entrenamiento con transfer learning (EfficientNetB0 preentrenada en ImageNet).

El modelo resultante es compatible con la app: entrada (48, 48, 1) con valores
0-255 y las mismas 7 clases. Internamente redimensiona la cara a IMG_SIZE x IMG_SIZE
y la convierte a 3 canales para la red preentrenada.

Uso:
    venv\\Scripts\\python.exe train_transfer.py              # entrena
    venv\\Scripts\\python.exe train_transfer.py --benchmark  # mide la velocidad y sale
    venv\\Scripts\\python.exe train_transfer.py --eval       # solo evalúa modelFEC_v3.h5

Fases:
  1. Base congelada: solo se entrena la cabeza de clasificación (pocas épocas).
  2. Fine-tuning de toda la red con una tasa de aprendizaje baja.
El mejor modelo (por precisión en validación) se guarda en modelFEC_v3.h5.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "images")
CACHE = os.path.join(DATA_DIR, "cache_48.npz")
OUT_MODEL = os.path.join(BASE_DIR, "modelFEC_v3.h5")
LOG_CSV = os.path.join(BASE_DIR, "train_transfer_log.csv")

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(CLASSES)

parser = argparse.ArgumentParser()
parser.add_argument("--img-size", type=int, default=96, help="tamaño de entrada de la red preentrenada")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--head-epochs", type=int, default=3)
parser.add_argument("--ft-epochs", type=int, default=25)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--model", default=OUT_MODEL, help="archivo .h5 donde se guarda / que se evalúa")
args = parser.parse_args()
OUT_MODEL = os.path.abspath(args.model)


# ---------------------------------------------------------------- datos
def load_split(split):
    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, split, name)
        for f in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))
            xs.append(img)
            ys.append(label)
    return np.array(xs, dtype=np.uint8)[..., None], np.array(ys, dtype=np.int64)


def load_data():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["xt"], d["yt"], d["xv"], d["yv"]
    print("Leyendo imágenes (solo la primera vez)...")
    xt, yt = load_split("train")
    xv, yv = load_split("validation")
    np.savez_compressed(CACHE, xt=xt, yt=yt, xv=xv, yv=yv)
    return xt, yt, xv, yv


augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(10 / 360, fill_mode="nearest"),
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    layers.RandomZoom(0.1, fill_mode="nearest"),
    layers.RandomContrast(0.2),
], name="augment")


def make_ds(x, y, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    ds = ds.batch(args.batch)

    def prep(xb, yb):
        xb = tf.cast(xb, tf.float32)
        if training:
            xb = tf.clip_by_value(augment(xb, training=True), 0.0, 255.0)
        return xb, tf.one_hot(yb, NUM_CLASSES)

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- modelo
def build_model():
    inputs = layers.Input(shape=(48, 48, 1))           # igual que la app: gris 48x48, 0-255
    x = layers.Resizing(args.img_size, args.img_size, interpolation="bilinear")(inputs)
    x = layers.Concatenate()([x, x, x])                 # gris -> 3 canales
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                             input_shape=(args.img_size, args.img_size, 3))
    base.trainable = False
    x = base(x, training=False)                         # BatchNorm en modo inferencia
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="FaceEmotion_v3"), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])


# ---------------------------------------------------------------- evaluación
def predict(model, x, tta=True):
    x = x.astype("float32")
    p = model.predict(x, batch_size=256, verbose=0)
    if tta:
        p = (p + model.predict(x[:, :, ::-1, :], batch_size=256, verbose=0)) / 2
    return p


def report(y, p, title):
    pred = p.argmax(1)
    print(f"\n=== {title}: precisión {np.mean(pred == y) * 100:.2f} % ===")
    print(f"{'clase':10s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'n':>5s}")
    for i, name in enumerate(CLASSES):
        tp = np.sum((pred == i) & (y == i))
        prec = tp / max(np.sum(pred == i), 1)
        rec = tp / max(np.sum(y == i), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"{name:10s} {prec:9.3f} {rec:7.3f} {f1:6.3f} {np.sum(y == i):5d}")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    print("matriz de confusión (filas = real):")
    print("          " + " ".join(f"{c[:5]:>5s}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:10s}" + " ".join(f"{v:5d}" for v in row))
    return np.mean(pred == y)


def evaluate(xv, yv):
    from tensorflow.keras.models import load_model
    orig = load_model(os.path.join(BASE_DIR, "modelFEC.h5"))
    report(yv, predict(orig, xv, tta=False), "Original (modelFEC.h5)")
    report(yv, predict(orig, xv, tta=True), "Original + TTA")
    best = load_model(OUT_MODEL)
    report(yv, predict(best, xv, tta=False), "v3 transfer learning")
    report(yv, predict(best, xv, tta=True), "v3 transfer learning + TTA")


# ---------------------------------------------------------------- main
def main():
    tf.keras.utils.set_random_seed(42)
    xt, yt, xv, yv = load_data()
    print(f"train {xt.shape}  val {xv.shape}")

    if args.eval:
        evaluate(xv, yv)
        return

    model, base = build_model()

    if args.benchmark:
        base.trainable = True
        compile_model(model, 1e-4)
        ds = make_ds(xt[:args.batch * 30], yt[:args.batch * 30], True)
        model.fit(ds.take(5), verbose=0)  # calentamiento
        t = time.time()
        model.fit(ds.skip(5), verbose=0)
        per_step = (time.time() - t) / 25
        steps = int(np.ceil(len(xt) / args.batch))
        print(f"img_size={args.img_size}: {per_step:.3f} s/paso -> ~{per_step * steps / 60:.1f} min por época (fine-tuning)")
        p = os.path.join(BASE_DIR, "_bench.h5")
        model.save(p)
        m2 = keras.models.load_model(p)
        print("guardar/cargar .h5 OK:", np.allclose(model.predict(xv[:4].astype('float32'), verbose=0),
                                                    m2.predict(xv[:4].astype('float32'), verbose=0), atol=1e-5))
        os.remove(p)
        return

    # Pesos por clase: raíz cuadrada de los pesos "balanced"
    counts = np.bincount(yt, minlength=NUM_CLASSES)
    cw = np.sqrt(len(yt) / (NUM_CLASSES * counts))
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("pesos por clase:", {CLASSES[i]: round(w, 2) for i, w in class_weight.items()})

    train_ds = make_ds(xt, yt, True)
    val_ds = make_ds(xv, yv, False)
    ckpt = keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1)
    csv = keras.callbacks.CSVLogger(LOG_CSV, append=True)
    if os.path.exists(LOG_CSV):
        os.remove(LOG_CSV)

    # Fase 1: solo la cabeza
    print("\n### Fase 1: base congelada")
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
              class_weight=class_weight, callbacks=[ckpt, csv], verbose=2)

    # Fase 2: fine-tuning completo con tasa de aprendizaje con decaimiento coseno
    print("\n### Fase 2: fine-tuning de toda la red")
    base.trainable = True
    steps = int(np.ceil(len(xt) / args.batch)) * args.ft_epochs
    lr = keras.optimizers.schedules.CosineDecay(1e-4, decay_steps=steps, alpha=0.01)
    compile_model(model, lr)
    early = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1)
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs + args.ft_epochs,
              initial_epoch=args.head_epochs, class_weight=class_weight,
              callbacks=[ckpt, csv, early], verbose=2)

    evaluate(xv, yv)
    print(f"\nModelo guardado en {OUT_MODEL}")


if __name__ == "__main__":
    main()
