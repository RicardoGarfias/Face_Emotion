# Contexto del proyecto: Face_Emotion

Detección de emociones faciales en imágenes y en tiempo real con la webcam.
Proyecto basado en el repositorio de David Revelo Luna (https://github.com/DavidReveloLuna/Face_Emotion).

## Cómo funciona
1. **Detección de rostros**: red SSD de OpenCV DNN (Caffe)
   - `face_detector/deploy.prototxt`
   - `face_detector/res10_300x300_ssd_iter_140000.caffemodel`
   - Umbral de confianza: 0.4
2. **Clasificación de emociones**: CNN de Keras `modelFEC.h5`
   - Entrada: `(None, 48, 48, 1)`, que es el rostro recortado en escala de grises de 48x48 (sin normalizar, valores 0-255)
   - Salida: `(None, 7)` → `['angry','disgust','fear','happy','neutral','sad','surprise']`
   - Entrenada en Google Colab (`FaceEmotion.ipynb`)

## Archivos
| Archivo | Descripción |
|---|---|
| `FaceEmotionApp.py` | **Interfaz gráfica (Tkinter)** con botón "Cargar imagen" y botón "Cámara en tiempo real" (iniciar/detener). Muestra recuadros, etiqueta y el top 3 de emociones en español. |
| `FaceEmotionVideo.py` | Script original: webcam en ventana de OpenCV (salir con `q`). |
| `FaceEmotionVideoBar.py` | Variante con gráfica de barras de las probabilidades. |
| `FaceEmotion.ipynb` | Notebook de entrenamiento. |
| `modelFEC.h5` | Modelo entrenado (binario). |

## Entorno
- Windows 11, Python 3.12 en `venv/`
- tensorflow 2.21, tf_keras 2.21, keras 3.15, opencv-python 4.14, pillow, imutils
- **Importante:** `modelFEC.h5` solo carga con Keras 2. Hay que poner
  `os.environ["TF_USE_LEGACY_KERAS"] = "1"` **antes** de importar tensorflow.
- Cámara: `cv2.VideoCapture(0, cv2.CAP_DSHOW)`

## Ejecutar
```
venv\Scripts\python.exe FaceEmotionApp.py
```

## Estado
- La detección en imágenes ya está probada: `angry.png` da angry y `image.png` da neutral.
- La interfaz abre correctamente.
- Falta probar a fondo el modo cámara.
