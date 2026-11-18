# Contexto del proyecto: Face_Emotion

Detección de emociones faciales en imágenes y en tiempo real con la webcam.
Proyecto basado en el repositorio de David Revelo Luna (https://github.com/DavidReveloLuna/Face_Emotion).

## Cómo funciona
1. **Detección de rostros**: red SSD de OpenCV DNN (Caffe)
   - `face_detector/deploy.prototxt`
   - `face_detector/res10_300x300_ssd_iter_140000.caffemodel`
   - Umbral de confianza: 0.4
2. **Clasificación de emociones**: modelo Keras `.h5`
   - Entrada: `(None, 48, 48, 1)`, que es el rostro recortado en escala de grises de 48x48 (sin normalizar, valores 0-255)
   - Salida: `(None, 7)` → `['angry','disgust','fear','happy','neutral','sad','surprise']`
   - Se usa `modelFEC_v3.h5` si existe; si no, `modelFEC.h5`
3. **TTA**: se promedia la predicción del rostro y la de su versión volteada (en un solo batch)
4. **Suavizado temporal** en video (`emotion_smoothing.py`): media móvil exponencial por rostro, `alpha=0.3`
5. El modelo se envuelve en `tf.function` (`reduce_retracing=True`). Así baja de ~145 ms a ~22 ms por frame con EfficientNet

## Modelos
| Archivo | Descripción | Val. accuracy |
|---|---|---|
| `modelFEC.h5` | CNN original desde cero | 64.86 % |
| `modelFEC_v3.h5` | EfficientNetB0 (ImageNet) + fine-tuning, entrenado en Colab T4 | 67.32 %, 67.95 % con TTA |

`modelFEC_v3.h5`: la capa de entrada recibe 48x48x1 (0-255) y adentro hace Resizing a 96x96 y concatena 3 canales.
Se entrenó con aumento de datos, pesos por clase (raíz de "balanced") y label smoothing 0.1,
con 3 épocas de cabeza congelada y luego fine-tuning con decaimiento coseno (lr 1e-4).

## Archivos
| Archivo | Descripción |
|---|---|
| `FaceEmotionApp.py` | **Interfaz gráfica (Tkinter)**: cargar imagen y cámara en tiempo real; top 3 de emociones en español. |
| `FaceEmotionVideo.py` | Webcam en una ventana de OpenCV (salir con `q`). |
| `FaceEmotionVideoBar.py` | Igual que el anterior, más una gráfica de barras de las probabilidades. |
| `emotion_smoothing.py` | Clase `EmotionSmoother` para el suavizado temporal. |
| `train_transfer.py` | Entrenamiento del modelo v3 (`--eval` evalúa, `--benchmark` mide la velocidad). |
| `FaceEmotion_v3_Colab.ipynb` | El mismo entrenamiento en Colab con GPU (lleva `train_transfer.py` incrustado). |
| `FaceEmotion.ipynb` | Notebook original de entrenamiento. |
| `images.rar` | Dataset (FER-2013); se descomprime en `images/` (ignorado por git). |

## Entorno
- Windows 11, Python 3.12 en `venv/` (ver `requirements.txt`)
- **Importante:** los `.h5` solo cargan con Keras 2. Hay que poner
  `os.environ["TF_USE_LEGACY_KERAS"] = "1"` **antes** de importar tensorflow.
- Cámara: `cv2.VideoCapture(0, cv2.CAP_DSHOW)`
- La PC tiene una RTX 4060, pero TensorFlow no usa la GPU en Windows nativo (necesitaría WSL2). Para entrenar se usa Colab.

## Estado
- La app y los scripts funcionan con `modelFEC_v3.h5` a ~30 FPS en CPU.
- Ideas pendientes: alinear el rostro, hacer fine-tuning con fotos propias, usar etiquetas FER+, un ensamble de modelos o WSL2 para entrenar con la GPU.
