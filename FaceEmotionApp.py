## Interfaz gráfica para detección de emociones ##
## - Botón para cargar una imagen y detectar emociones
## - Botón para detección en tiempo real con la cámara

import os
# Debe definirse ANTES de importar tensorflow para usar Keras 2 (tf_keras)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model

from emotion_smoothing import EmotionSmoother

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tipos de emociones del detector
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASSES_ES = {
    'angry': 'Enojo', 'disgust': 'Asco', 'fear': 'Miedo', 'happy': 'Felicidad',
    'neutral': 'Neutral', 'sad': 'Tristeza', 'surprise': 'Sorpresa',
}

# Tamaño máximo del área de visualización
VIEW_W, VIEW_H = 800, 600


def emotion_model_paths():
    """Modelos de emociones a usar.

    Si hay modelos v4 (modelFEC_v4*.h5), se usan todos junto con el v3 como ensamble
    (se promedian sus predicciones). Si no, el v3. Si tampoco existe, el original.
    """
    names = sorted(f for f in os.listdir(BASE_DIR) if f.startswith("modelFEC_v4") and f.endswith(".h5"))
    if os.path.exists(os.path.join(BASE_DIR, "modelFEC_v3.h5")):
        names.append("modelFEC_v3.h5")
    return [os.path.join(BASE_DIR, n) for n in (names or ["modelFEC.h5"])]


def fast_predictor(models):
    """Compila el modelo (o el ensamble) con tf.function: ~6x más rápido que llamarlo directo en cada frame."""
    fn = tf.function(lambda x: tf.add_n([m(x, training=False) for m in models]) / len(models),
                     reduce_retracing=True)
    for n in (2, 4):  # calentamiento para que el primer frame no se trabe
        fn(tf.zeros((n, 48, 48, 1)))
    return lambda x, training=False: fn(tf.convert_to_tensor(x, tf.float32))


def load_models():
    prototxt = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
    weights = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    face_net = cv2.dnn.readNet(prototxt, weights)
    emotion_model = fast_predictor([load_model(p, compile=False) for p in emotion_model_paths()])
    return face_net, emotion_model


def predict_emotion(frame, face_net, emotion_model, threshold=0.4):
    """Devuelve las ubicaciones de los rostros y las predicciones de emoción."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs = [], []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            xi, yi, xf, yf = box.astype("int")
            xi, yi = max(0, xi), max(0, yi)
            xf, yf = min(w, xf), min(h, yf)

            face = frame[yi:yf, xi:xf]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48)).astype("float32")
            faces.append(face[..., np.newaxis])
            locs.append((xi, yi, xf, yf))

    # Ordena los rostros en orden de lectura (por filas, de izquierda a derecha)
    # para que "Rostro 1, 2, 3..." coincida con lo que se ve en la imagen
    order = reading_order(locs)
    faces = [faces[i] for i in order]
    locs = [locs[i] for i in order]

    preds = []
    if faces:
        # Test-time augmentation: se predice el rostro original y el volteado
        # horizontalmente en una sola llamada al modelo, y se promedian
        batch = np.array(faces)
        both = np.concatenate([batch, batch[:, :, ::-1, :]])
        out = emotion_model(both, training=False).numpy()
        preds = (out[:len(faces)] + out[len(faces):]) / 2
    return locs, preds


def reading_order(locs):
    """Índices de las cajas ordenadas por filas (arriba -> abajo) y luego de izquierda a derecha."""
    if not locs:
        return []
    by_y = sorted(range(len(locs)), key=lambda i: (locs[i][1] + locs[i][3]) / 2)
    rows, current = [], [by_y[0]]
    for i in by_y[1:]:
        ref = locs[current[0]]
        cy = (locs[i][1] + locs[i][3]) / 2
        if abs(cy - (ref[1] + ref[3]) / 2) < (ref[3] - ref[1]) / 2:  # misma fila
            current.append(i)
        else:
            rows.append(current)
            current = [i]
    rows.append(current)
    return [i for row in rows for i in sorted(row, key=lambda i: locs[i][0])]


def draw_results(frame, locs, preds):
    h, w = frame.shape[:2]
    for (xi, yi, xf, yf), pred in zip(locs, preds):
        idx = int(np.argmax(pred))
        label = "{}: {:.0f}%".format(CLASSES[idx], pred[idx] * 100)
        # El tamaño del texto depende del tamaño del rostro, no de la imagen completa
        scale = float(np.clip((xf - xi) / 160, 0.5, 1.6))
        thick = 2 if scale >= 0.75 else 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, thick)
        pad = int(6 * scale) + 2
        # La etiqueta va arriba del recuadro, sin salirse de la imagen
        lx = int(np.clip(xi, 0, max(w - tw - 2 * pad, 0)))
        top = max(yi - th - 2 * pad, 0)
        cv2.rectangle(frame, (lx, top), (lx + tw + 2 * pad, top + th + 2 * pad), (255, 0, 0), -1)
        cv2.putText(frame, label, (lx + pad, top + th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (0, 255, 0), thick, cv2.LINE_AA)
        cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 0, 0), max(2, int(round(3 * scale))))
    return frame


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Emociones")
        self.root.configure(bg="#1e1e2e")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam = None
        self.running = False
        self.prev_time = 0
        self.photo = None
        self.smoother = EmotionSmoother()

        # --- Barra de botones ---
        bar = tk.Frame(root, bg="#1e1e2e")
        bar.pack(pady=12)

        btn_style = dict(font=("Segoe UI", 12, "bold"), fg="white", bd=0,
                         padx=18, pady=10, cursor="hand2", activeforeground="white")

        self.btn_img = tk.Button(bar, text="📁  Cargar imagen", bg="#3b82f6",
                                 activebackground="#2563eb", command=self.load_image, **btn_style)
        self.btn_img.pack(side=tk.LEFT, padx=8)

        self.btn_cam = tk.Button(bar, text="📷  Cámara en tiempo real", bg="#10b981",
                                 activebackground="#059669", command=self.toggle_camera, **btn_style)
        self.btn_cam.pack(side=tk.LEFT, padx=8)

        # --- Área de imagen ---
        view = tk.Frame(root, bg="#11111b", width=VIEW_W, height=VIEW_H)
        view.pack(padx=12)
        view.pack_propagate(False)  # tamaño fijo en píxeles
        self.canvas = tk.Label(view, bg="#11111b",
                               text="Carga una imagen o inicia la cámara",
                               fg="#a6adc8", font=("Segoe UI", 14))
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Resultados ---
        self.result = tk.Label(root, text="", bg="#1e1e2e", fg="#cdd6f4",
                               font=("Segoe UI", 12), justify=tk.LEFT)
        self.result.pack(pady=10)

        self.status = tk.Label(root, text="Cargando modelos...", bg="#1e1e2e",
                               fg="#f9e2af", font=("Segoe UI", 10))
        self.status.pack(pady=(0, 8))
        self.set_buttons(tk.DISABLED)
        self.root.update()

        try:
            self.face_net, self.emotion_model = load_models()
            names = [os.path.basename(p) for p in emotion_model_paths()]
            label = names[0] if len(names) == 1 else f"ensamble: {' + '.join(names)}"
            self.status.config(text=f"Modelos listos ({label})", fg="#a6e3a1")
            self.set_buttons(tk.NORMAL)
        except Exception as e:
            self.status.config(text="Error cargando modelos", fg="#f38ba8")
            messagebox.showerror("Error", f"No se pudieron cargar los modelos:\n{e}")

    def set_buttons(self, state):
        self.btn_img.config(state=state)
        self.btn_cam.config(state=state)

    # ---------- Utilidades de visualización ----------
    def show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((VIEW_W, VIEW_H))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo, text="")

    def show_summary(self, preds):
        if len(preds) == 0:
            self.result.config(text="No se detectaron rostros")
            return
        lines = []
        for n, pred in enumerate(preds, 1):
            idx = int(np.argmax(pred))
            top3 = np.argsort(pred)[::-1][:3]
            detail = ", ".join(f"{CLASSES_ES[CLASSES[j]]} {pred[j]*100:.0f}%" for j in top3)
            lines.append(f"Rostro {n}: {CLASSES_ES[CLASSES[idx]]}   ({detail})")
        self.result.config(text="\n".join(lines))

    # ---------- Modo imagen ----------
    def load_image(self):
        if self.running:
            self.stop_camera()
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Todos", "*.*")])
        if not path:
            return
        # imdecode soporta rutas con acentos/espacios en Windows
        data = np.fromfile(path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return

        locs, preds = predict_emotion(frame, self.face_net, self.emotion_model)
        self.show_frame(draw_results(frame, locs, preds))
        self.show_summary(preds)
        self.status.config(text=os.path.basename(path), fg="#a6e3a1")

    # ---------- Modo cámara ----------
    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            self.cam = None
            return
        self.running = True
        self.smoother.reset()
        self.prev_time = time.time()
        self.btn_cam.config(text="⏹  Detener cámara", bg="#ef4444", activebackground="#dc2626")
        self.status.config(text="Cámara activa", fg="#a6e3a1")
        self.update_camera()

    def stop_camera(self):
        self.running = False
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self.btn_cam.config(text="📷  Cámara en tiempo real", bg="#10b981", activebackground="#059669")
        self.status.config(text="Cámara detenida", fg="#f9e2af")

    def update_camera(self):
        if not self.running:
            return
        ret, frame = self.cam.read()
        if ret:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
            locs, preds = predict_emotion(frame, self.face_net, self.emotion_model)
            # Promedia con los frames anteriores para que la etiqueta no parpadee
            preds = self.smoother.update(locs, preds)
            draw_results(frame, locs, preds)

            now = time.time()
            fps = 1.0 / (now - self.prev_time) if now > self.prev_time else 0
            self.prev_time = now
            cv2.putText(frame, f"{int(fps)} FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 3, cv2.LINE_AA)

            self.show_frame(frame)
            self.show_summary(preds)
        self.root.after(10, self.update_camera)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    EmotionApp(root)
    root.mainloop()
