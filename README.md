# **Face_Emotion**

Detección de emociones faciales en imágenes y en tiempo real con la webcam, usando una red neuronal convolucional.

Basado en el repositorio de [David Revelo Luna](https://github.com/DavidReveloLuna/Face_Emotion).

<img src="assets/ToGift.gif" width="500">

## Emociones detectadas

| Clase | Emoción |
|---|---|
| `angry` | Enojo |
| `disgust` | Asco |
| `fear` | Miedo |
| `happy` | Felicidad |
| `neutral` | Neutral |
| `sad` | Tristeza |
| `surprise` | Sorpresa |

## Cómo funciona

1. **Detección de rostros:** red SSD de OpenCV DNN ([face_detector/](face_detector/)).
2. **Clasificación de emociones:** cada rostro se convierte a escala de grises de 48×48 y se clasifica con una CNN.
3. **Test-time augmentation:** se promedia la predicción del rostro original y la del volteado horizontalmente.
4. **Suavizado temporal** (en video): las probabilidades de cada rostro se promedian entre frames para que la etiqueta no parpadee ([emotion_smoothing.py](emotion_smoothing.py)).

## Modelos

| Archivo | Descripción | Precisión en validación |
|---|---|---|
| `modelFEC.h5` | CNN original entrenada desde cero ([FaceEmotion.ipynb](FaceEmotion.ipynb)) | 64.86 % |
| `modelFEC_v3.h5` | EfficientNetB0 preentrenada en ImageNet + fine-tuning ([train_transfer.py](train_transfer.py)) | **67.95 %** (con TTA) |

La app y los scripts usan `modelFEC_v3.h5` si existe y, si no, el original.

Precisión por emoción (F1) con `modelFEC_v3.h5`: felicidad 0.87, sorpresa 0.79, asco 0.70, neutral 0.63, enojo 0.60, tristeza 0.55, miedo 0.51.

## Instalación (Windows, Python 3.12)

    > python -m venv venv
    > venv\Scripts\activate
    > pip install -r requirements.txt

> Los modelos `.h5` se guardaron con Keras 2. Los scripts definen `TF_USE_LEGACY_KERAS=1`
> antes de importar tensorflow para usar `tf_keras`.

## Uso

**Interfaz gráfica:** permite cargar una imagen o usar la cámara, y muestra el top 3 de emociones en español.

    > python FaceEmotionApp.py

**Webcam en ventana de OpenCV** (salir con `q`):

    > python FaceEmotionVideo.py

**Webcam con gráfica de barras** de las probabilidades:

    > python FaceEmotionVideoBar.py

## Entrenamiento

El dataset (FER-2013, 28,821 imágenes de entrenamiento y 7,066 de validación) está en `images.rar`.

**En Google Colab con GPU (~30-40 min, recomendado):** sube [FaceEmotion_v3_Colab.ipynb](FaceEmotion_v3_Colab.ipynb) a Colab,
activa la GPU T4 y ejecuta todo. Al final se descarga `modelFEC_v3.h5`.

**En la PC con CPU (~2-3 h):**

    > "C:\Program Files\WinRAR\UnRAR.exe" x images.rar
    > python train_transfer.py              # entrena y guarda modelFEC_v3.h5
    > python train_transfer.py --eval       # compara con el modelo original

El notebook original del autor, con la CNN entrenada desde cero, es [FaceEmotion.ipynb](FaceEmotion.ipynb).

# **Canal de Youtube del autor original**
[Click aquí para ver el canal de YOUTUBE](https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg)
