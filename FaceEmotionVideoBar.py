## Detección de emociones en tiempo real (con gráfica de barras) ##
## SISTEMAS INTELIGENTES ##
## https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg
## David Revelo Luna

import os
# Debe definirse ANTES de importar tensorflow para usar Keras 2 (tf_keras),
# que es el único con el que carga modelFEC.h5
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import de librerias
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time

from emotion_smoothing import EmotionSmoother

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tipos de emociones del detector
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
my_colors = ['r', 'g', 'b', 'y', 'k', 'm', 'c']

# Gráfica de barras interactiva: se crea una vez y solo se actualizan las alturas
plt.ion()
figura1, ax = plt.subplots()
x = np.arange(len(classes))
bars = ax.bar(x, np.zeros(len(classes)), color=my_colors, width=1)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim([0.0, 1.0])
ax.grid(True)
ax.set_title("Probabilidad por emoción")
figura1.show()

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = time.time()
fps = 0

# Cargamos el modelo de detección de rostros
prototxtPath = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
weightsPath = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
# Usa el modelo v3 (transfer learning) si existe; si no, el original
modelPath = os.path.join(BASE_DIR, "modelFEC_v3.h5")
if not os.path.exists(modelPath):
    modelPath = os.path.join(BASE_DIR, "modelFEC.h5")
print("Modelo de emociones:", os.path.basename(modelPath))
emotionModel = load_model(modelPath, compile=False)
# Compila el modelo con tf.function: ~6x más rápido que llamarlo directo en cada frame
_fast = tf.function(lambda x: emotionModel(x, training=False), reduce_retracing=True)
smoother = EmotionSmoother()


# Toma la imagen, los modelos de detección de rostros y de emociones
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame, faceNet, emotionModel):
    h, w = frame.shape[:2]
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida que el bounding box quede dentro de la imagen
            Xi, Yi = max(0, Xi), max(0, Yi)
            Xf, Yf = min(w, Xf), min(h, Yf)

            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48)).astype("float32")

            faces.append(face[..., np.newaxis])
            locs.append((Xi, Yi, Xf, Yf))

    preds = []
    if faces:
        # Test-time augmentation: original + volteado horizontal, promediados
        batch = np.array(faces)
        out = _fast(tf.convert_to_tensor(np.concatenate([batch, batch[:, :, ::-1, :]]))).numpy()
        preds = (out[:len(faces)] + out[len(faces):]) / 2
    return (locs, preds)


def update_bars(probs):
    for bar, p in zip(bars, probs):
        bar.set_height(float(p))
    figura1.canvas.draw_idle()
    figura1.canvas.flush_events()


# Se crea la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    raise SystemExit("No se pudo abrir la cámara.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("No se pudo leer un frame de la cámara.")
        break
    frame = imutils.resize(frame, width=640)

    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
    # Promedia con los frames anteriores para que la etiqueta no parpadee
    preds = smoother.update(locs, preds)

    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        idx = int(np.argmax(pred))
        label = "{}: {:.0f}%".format(classes[idx], pred[idx] * 100)

        top = max(Yi - 40, 0)
        cv2.rectangle(frame, (Xi, top), (Xf, top + 40), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi + 5, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

    # La gráfica muestra el primer rostro detectado (o ceros si no hay)
    update_bars(preds[0] if len(preds) else np.zeros(len(classes)))

    time_actualframe = time.time()
    if time_actualframe > time_prevframe:
        fps = 1 / (time_actualframe - time_prevframe)
    time_prevframe = time_actualframe

    cv2.putText(frame, str(int(fps)) + " FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    # Salir con 'q' o cerrando la ventana de la gráfica
    if cv2.waitKey(1) & 0xFF == ord('q') or not plt.fignum_exists(figura1.number):
        break

cv2.destroyAllWindows()
cam.release()
plt.close('all')
