## Detección de emociones en tiempo real ##
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = time.time()
fps = 0

# Tipos de emociones del detector
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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

    # Listas para guardar rostros y ubicaciones
    faces = []
    locs = []

    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):
        # Umbral para determinar que la detección es confiable
        if detections[0, 0, i, 2] > 0.4:
            # Bounding box escalado a las dimensiones de la imagen
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida que el bounding box quede dentro de la imagen
            Xi, Yi = max(0, Xi), max(0, Yi)
            Xf, Yf = min(w, Xf), min(h, Yf)

            # Se extrae el rostro, se convierte a gris y se escala a 48x48
            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48)).astype("float32")

            faces.append(face[..., np.newaxis])
            locs.append((Xi, Yi, Xf, Yf))

    preds = []
    if faces:
        # Una sola llamada al modelo para todos los rostros
        # Test-time augmentation: original + volteado horizontal, promediados
        batch = np.array(faces)
        out = _fast(tf.convert_to_tensor(np.concatenate([batch, batch[:, :, ::-1, :]]))).numpy()
        preds = (out[:len(faces)] + out[len(faces):]) / 2
    return (locs, preds)


# Se crea la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    raise SystemExit("No se pudo abrir la cámara.")

while True:
    # Se toma un frame de la cámara y se redimensiona
    ret, frame = cam.read()
    if not ret:
        print("No se pudo leer un frame de la cámara.")
        break
    frame = imutils.resize(frame, width=640)

    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
    # Promedia con los frames anteriores para que la etiqueta no parpadee
    preds = smoother.update(locs, preds)

    # Para cada hallazgo se dibuja en la imagen el bounding box y la clase
    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        idx = int(np.argmax(pred))
        label = "{}: {:.0f}%".format(classes[idx], pred[idx] * 100)

        top = max(Yi - 40, 0)
        cv2.rectangle(frame, (Xi, top), (Xf, top + 40), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi + 5, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

    time_actualframe = time.time()
    if time_actualframe > time_prevframe:
        fps = 1 / (time_actualframe - time_prevframe)
    time_prevframe = time_actualframe

    cv2.putText(frame, str(int(fps)) + " FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
