"""Suavizado temporal de predicciones de emoción para video.

Cada rostro detectado se asocia con el rostro más cercano del frame anterior
(por distancia entre centros) y sus probabilidades se promedian con una media
móvil exponencial. Así la etiqueta no "parpadea" entre emociones de un frame
a otro.
"""
import numpy as np


class EmotionSmoother:
    def __init__(self, alpha=0.3, max_dist=80, max_missing=10):
        # alpha: peso del frame nuevo (menor = más suave pero reacciona más lento)
        # max_dist: distancia máxima en píxeles entre centros para considerar que es el mismo rostro
        # max_missing: frames que se conserva un rostro que dejó de detectarse
        self.alpha = alpha
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.tracks = []  # cada track: {"center": (x, y), "probs": array, "missing": int}

    def reset(self):
        self.tracks = []

    def update(self, locs, preds):
        """Recibe las cajas y predicciones del frame actual y devuelve las predicciones suavizadas."""
        centers = [((xi + xf) / 2, (yi + yf) / 2) for (xi, yi, xf, yf) in locs]
        used = set()
        smoothed = []

        for center, pred in zip(centers, preds):
            pred = np.asarray(pred, dtype="float32")
            best, best_d = None, self.max_dist
            for k, t in enumerate(self.tracks):
                if k in used:
                    continue
                d = np.hypot(center[0] - t["center"][0], center[1] - t["center"][1])
                if d < best_d:
                    best, best_d = k, d

            if best is None:
                self.tracks.append({"center": center, "probs": pred, "missing": 0})
                used.add(len(self.tracks) - 1)
                smoothed.append(pred)
            else:
                t = self.tracks[best]
                t["probs"] = self.alpha * pred + (1 - self.alpha) * t["probs"]
                t["center"] = center
                t["missing"] = 0
                used.add(best)
                smoothed.append(t["probs"])

        # Envejece los rostros que no aparecieron en este frame
        for k, t in enumerate(self.tracks):
            if k not in used:
                t["missing"] += 1
        self.tracks = [t for t in self.tracks if t["missing"] <= self.max_missing]

        return np.array(smoothed)
