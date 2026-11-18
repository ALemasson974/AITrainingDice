# `dice_reader.py` : explication détaillée

> Explication du script [`scripts/dice_reader.py`](../scripts/dice_reader.py),
> la démo finale du projet. Les numéros de ligne correspondent à la version
> actuelle du script : s'ils changent après une modification, mettez-les à jour ici.

## 1. À quoi sert ce script ?

C'est le **programme final** : il filme la table avec la webcam et affiche en direct :
- un cadre et une étiquette sur chaque dé, par exemple `D20: 17` ;
- à droite, un panneau avec la liste des dés (de gauche à droite), la valeur de chacun et le **total**.

Pour cela il enchaîne **deux modèles** : c'est le pipeline « en deux étapes », l'option (b) de [`model_B_notes.md`](model_B_notes.md).

```
  Image de la webcam (640×480)
        │
        ▼
  Modèle A : détecteur YOLO  (models/type_best.pt)
  « où sont les dés et de quel type ? »   → un cadre + un type par dé
        │
        ▼
  On découpe chaque cadre dans l'image  → une petite image par dé (un « crop »)
        │
        ▼  (tous les crops en un seul appel)
  Modèle B : classifieur YOLO  (models/cls_best.pt)
  « quel nombre est sur ce dé ? »         → une face par dé
        │
        ▼
  Lissage : chaque dé est suivi d'image en image, et on garde
  le vote majoritaire de ses 10 dernières lectures
        │
        ▼
  Fenêtre : cadres + « D20: 17 » + liste + TOTAL
```

| | Modèle A | Modèle B |
|---|---|---|
| Fichier | `models/type_best.pt` (notre run 2) | `models/cls_best.pt` (entraîné par un coéquipier) |
| Type de modèle | **détecteur** : renvoie des cadres | **classifieur** : une seule réponse pour toute l'image |
| Classes | D6, D8, D10, D12, D20 | 21 faces : « 0 » à « 20 » |
| Taille d'image | 416 | 128 |

Le script est **construit à partir du modèle du cours** `webcam_detection.py` : mêmes options (`--source`, `--conf`, `--imgsz`), même boucle caméra, mêmes touches et même gestion des erreurs. Le modèle du cours n'est pas modifié.

---

## 2. Les imports ([lignes 27-36](../scripts/dice_reader.py#L27-L36))

```python
from __future__ import annotations   # permet d'écrire "-> int" etc. sur les vieilles versions de Python
import argparse                      # options de la ligne de commande
from collections import Counter, deque   # compter des votes, file de taille fixe (pour le lissage)
from datetime import datetime        # date/heure, pour nommer les captures d'écran
from pathlib import Path             # chemins de fichiers
import cv2                           # OpenCV : caméra, dessin, fenêtre
import numpy as np                   # NumPy : une image est un tableau de nombres
from ultralytics import YOLO         # charger et utiliser les modèles
```

**Une image, c'est un tableau NumPy.** OpenCV lit une image de la webcam sous la forme d'un tableau de forme `(hauteur, largeur, 3)`, par exemple `(480, 640, 3)`. Chaque pixel a 3 valeurs de 0 à 255, dans l'ordre **Bleu, Vert, Rouge** (BGR), et non RGB comme ailleurs.

---

## 3. Les constantes ([lignes 38-70](../scripts/dice_reader.py#L38-L70))

### 3.1 `FACES_BY_TYPE` : les faces possibles de chaque dé

```python
FACES_BY_TYPE = {
    "D6": {"1", "2", "3", "4", "5", "6"},
    ...
    "D10": {"0", "1", ..., "9"},
    "D12": {str(number) for number in range(1, 13)},   # "1" ... "12"
    "D20": {str(number) for number in range(1, 21)},   # "1" ... "20"
}
```
- C'est un **dictionnaire** (≈ `HashMap`) dont chaque valeur est un **ensemble** (`set`, ≈ `HashSet`) de faces écrites en texte.
- `{str(number) for number in range(1, 13)}` est une **compréhension d'ensemble** : elle construit `{"1", "2", …, "12"}` en une ligne. `range(1, 13)` va de 1 à 12, car la borne de fin est exclue, comme dans une boucle `for (i = 1; i < 13; i++)` en Java.

**À quoi ça sert ?** Le modèle B lit le nombre **sans savoir** de quel dé il s'agit. Il peut donc répondre « 8 » sur un dé que le modèle A a identifié comme D6, ce qui est impossible : l'un des deux modèles se trompe. Ce tableau permet de **détecter** cette incohérence, et le dé est alors affiché en orange avec un « ? ».

**Pourquoi ne pas forcer le modèle B à choisir une face possible ?** On l'a testé sur les 375 dés annotés : forcer donnait **2 erreurs de plus** (95,7 % au lieu de 96,3 %). Quand la combinaison est impossible, c'est en général le **type** (modèle A) qui est faux, par exemple un D8 vu comme un D6. Forcer transformait alors un nombre juste en nombre faux. On signale donc l'incohérence au lieu de la « corriger ».

### 3.2 Couleurs et panneau

```python
COLOR_SURE = (80, 200, 80)       # vert : lecture fiable
COLOR_UNSURE = (0, 165, 255)     # orange : à vérifier à l'œil
PANEL_WIDTH = 270                # largeur du panneau de droite, en pixels
```
Les couleurs sont en **(Bleu, Vert, Rouge)** : `(0, 165, 255)` = beaucoup de rouge et un peu de vert, donc orange. Les constantes sont écrites en MAJUSCULES par convention, comme les `static final` en Java ; Python n'empêche pas de les modifier.

### 3.3 Réglages du lissage

```python
MIN_SEEN_FRAMES = 3      # un nouveau dé n'apparaît qu'après avoir été vu dans 3 images
MAX_MISSED_FRAMES = 5    # un dé non trouvé reste affiché jusqu'à 5 images (~0,15 s)
MAX_JUMP = 0.5           # « même dé » si le centre a bougé de moins d'une demi-taille de dé
```
Le rôle de chacun est expliqué en 4.4.

---

## 4. Les fonctions

### 4.1 `face_value` : la face devient une valeur ([lignes 73-80](../scripts/dice_reader.py#L73-L80))

```python
def face_value(face_name):
    value = int(face_name)      # "17" -> 17
    if value == 0:
        return 10
    return value
```
`int("17")` convertit le texte en nombre, comme `Integer.parseInt` en Java. **Le « 0 » du D10 vaut 10.** Seul le D10 a une face « 0 », donc un « 0 » vaut toujours 10, même si le modèle A s'est trompé de type.

### 4.2 `crop_die` : découper un dé dans l'image ([lignes 83-99](../scripts/dice_reader.py#L83-L99))

```python
frame_height, frame_width = frame.shape[:2]        # (480, 640)
x1, y1, x2, y2 = box_xyxy                          # coins du cadre, en pixels
extra_x = (x2 - x1) * margin                       # marge éventuelle autour du cadre
...
left = max(0, int(x1 - extra_x))                   # max/min : ne pas sortir de l'image
...
return frame[top:bottom, left:right]
```
- `frame.shape[:2]` garde les 2 premiers éléments de `(480, 640, 3)`, donc `(480, 640)`. `[:2]` est une **tranche** (*slice*) : « du début jusqu'à l'indice 2 exclu ».
- `x1, y1, x2, y2 = box_xyxy` est un **déballage** (*unpacking*) : les 4 valeurs de la liste vont dans 4 variables d'un coup.
- `frame[top:bottom, left:right]` découpe un rectangle : les lignes `top` à `bottom-1` et les colonnes `left` à `right-1`, comme `substring` en Java mais en 2 dimensions. Le résultat est une nouvelle petite image : le **crop** du dé.

**La marge (`--margin`, 0 par défaut).** On peut agrandir le cadre avant de découper. Mais le modèle B doit recevoir des crops **découpés comme ceux de son entraînement**. On a mesuré sur les 375 dés annotés :

| Marge | Dés lus correctement |
|---|---|
| **0 (le cadre exact)** | **96,3 %** |
| 10 % | 95,2 % |
| 15 % | 93,6 % |
| 25 % | 79,5 % |

Le coéquipier a donc entraîné le modèle B sur des crops **serrés**. Avec une grande marge, des morceaux de table et des dés voisins apparaissent dans le crop, et le modèle lit moins bien.

### 4.3 `read_dice` : le cœur du pipeline ([lignes 102-140](../scripts/dice_reader.py#L102-L140))

**Étape 1 : le modèle A trouve les dés.**
```python
detection = type_model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device="cpu", verbose=False)[0]
boxes = detection.boxes.xyxy.tolist()               # [[x1, y1, x2, y2], ...]
type_indices = detection.boxes.cls.tolist()         # [4.0, 0.0, ...]
type_confidences = detection.boxes.conf.tolist()    # [0.91, 0.87, ...]
```
- `predict` renvoie une liste de résultats, un par image ; on n'en donne qu'une, d'où le `[0]`.
- `conf=0.25` : les cadres dont la confiance est inférieure à 25 % sont ignorés.
- `.boxes.xyxy` donne les coins des cadres en pixels, `.boxes.cls` le numéro de classe (4 = D20) et `.boxes.conf` la confiance. Ce sont des **tenseurs** PyTorch ; `.tolist()` les transforme en listes Python normales.
- S'il n'y a aucun dé, on renvoie une liste vide `[]`.

**Étape 2 : découper tous les dés et les lire en un seul appel.**
```python
crops = [crop_die(frame, box, args.margin) for box in boxes]
readings = number_model.predict(crops, imgsz=args.cls_imgsz, device="cpu", verbose=False)
```
- `[... for box in boxes]` est une **compréhension de liste** : une liste de crops, un par dé. C'est l'équivalent d'une boucle `for` avec `list.add(...)`.
- On donne **toute la liste** au modèle B en un seul appel, ce qui est plus rapide que 5 appels séparés. Il renvoie une lecture par crop, dans le même ordre.

**Étape 3 : assembler les informations de chaque dé.**
```python
for box, type_index, type_conf, reading in zip(boxes, type_indices, type_confidences, readings):
    die_type = type_model.names[int(type_index)]          # 4 -> "D20"
    face = number_model.names[reading.probs.top1]         # indice -> "17"
    dice.append({
        "type": die_type,
        "face": face,
        "value": face_value(face),
        "type_conf": type_conf,
        "face_conf": float(reading.probs.top1conf),
        "consistent": face in FACES_BY_TYPE[die_type],
        "box": box,
    })
```
- `zip(a, b, c, d)` parcourt plusieurs listes **en même temps** : au 1er tour, le 1er élément de chacune, et ainsi de suite.
- `model.names` est un dictionnaire `{indice: nom}` qui transforme un numéro de classe en nom lisible.
- `reading.probs.top1` est l'indice de la face la plus probable, et `top1conf` sa probabilité (0 à 1).
- `face in FACES_BY_TYPE[die_type]` vaut `True` si la face existe sur ce type de dé.
- Chaque dé est rangé dans un **dictionnaire**, à peu près un petit objet sans classe.

**Étape 4 : trier de gauche à droite.**
```python
dice.sort(key=lambda die: die["box"][0])
```
`lambda die: die["box"][0]` est une **petite fonction anonyme**, comme une lambda Java `die -> die.box[0]`. Elle renvoie le `x1` du cadre, donc on trie par position horizontale. La liste du panneau suit ainsi l'ordre des dés sur la table.

### 4.4 `Track` et `DiceTracker` : le lissage ([lignes 143-223](../scripts/dice_reader.py#L143-L223))

**Le problème.** Chaque image est analysée indépendamment. Une lecture peut donc « clignoter » : un 17 lu 12 pendant une seule image, un dé oublié une image, un faux cadre qui apparaît une fraction de seconde. Le total à l'écran saute alors sans arrêt.

**L'idée.** On ne montre pas la lecture de l'image actuelle, mais le **vote majoritaire des 10 dernières lectures** de chaque dé (`--smooth-frames 10`). Pour voter « par dé », il faut savoir que le D20 de cette image est **le même** que le D20 de l'image précédente. Le modèle A ne numérote pas les dés : il les retrouve à chaque image. On fait donc un **suivi** (*tracking*) simple : sur une table, un dé ne bouge presque pas entre deux images (~33 ms d'écart), donc on l'associe au dé le plus proche de l'image précédente.

**La classe `Track`** : un dé suivi d'image en image. C'est une petite classe, comme une classe Java avec des attributs publics.
```python
class Track:
    def __init__(self, history_size):          # __init__ = le constructeur
        self.box = None                        # self = this en Java
        self.center = None
        self.history = deque(maxlen=history_size)   # ses N dernières lectures
        self.frames_seen = 0
        self.frames_missed = 0
```
`deque(maxlen=10)` est une **file de taille fixe** : quand on ajoute une 11ᵉ lecture, la plus ancienne est retirée automatiquement. On garde donc toujours les 10 plus récentes.

**`DiceTracker.update(dice)`**, appelée à chaque image, fait 3 choses :

1. **Associer chaque dé détecté à un dé déjà suivi.**
   ```python
   distance = ((center[0] - track.center[0]) ** 2 + (center[1] - track.center[1]) ** 2) ** 0.5
   if distance <= max_distance and (best_distance is None or distance < best_distance):
   ```
   - On calcule la distance entre les centres (Pythagore ; `** 0.5` = racine carrée) et on garde le dé suivi **le plus proche**, s'il est à moins d'une demi-taille de dé (`MAX_JUMP`).
   - Si aucun n'est assez proche, c'est un **nouveau dé** : on crée un `Track`.
   - Chaque dé suivi ne peut être associé qu'une fois par image (`tracks_not_matched.remove(...)`).
   - On ajoute la lecture `(type, face)` à son historique.
2. **Oublier les dés disparus.** Un dé suivi qui n'a pas été trouvé dans cette image voit `frames_missed` augmenter. Au-delà de 5 images de suite (`MAX_MISSED_FRAMES`), il est oublié. Avant cela, il reste affiché avec son dernier résultat, ce qui évite qu'un dé clignote quand le modèle A le rate une fois.
3. **Voter.**
   ```python
   face_votes = Counter(face for _die_type, face in track.history)
   face, votes = face_votes.most_common(1)[0]
   ```
   - `Counter` compte combien de fois chaque valeur apparaît : `Counter(["17", "17", "12"])` donne `{"17": 2, "12": 1}`.
   - `most_common(1)` donne le gagnant et son nombre de voix : `[("17", 2)]`.
   - On vote séparément pour le **type** et pour la **face**.
   - Un dé n'est affiché qu'après avoir été vu dans **3 images** (`MIN_SEEN_FRAMES`) : un faux cadre d'une seule image n'apparaît jamais.
   - Le pourcentage affiché devient la **part des images qui sont d'accord** avec le vote (80 % = 8 des 10 dernières).

**Test sur une séquence contrôlée de 15 images :**
- les deux mauvaises lectures (« 12 » au lieu de « 17 ») n'apparaissent jamais ;
- le D10 oublié une image reste affiché ;
- le faux D6 d'une seule image n'est jamais montré ;
- quand le D20 passe à 5, l'écran change après 5 images (~0,2 s). Pendant la transition, il est en orange (50 %), ce qui signale que la lecture est en train de changer.

**Le compromis.** Plus `--smooth-frames` est grand, plus l'affichage est stable, mais plus il met de temps à suivre un dé qu'on vient de relancer. 10 images ≈ 0,3 s à 30 images/s.

**Limite.** Un dé déplacé **très vite** d'un endroit à un autre est vu comme un nouveau dé. Son ancienne position reste affichée jusqu'à 5 images, donc pendant un court instant il peut apparaître deux fois. Un dé qui roule n'est pas affiché tant qu'il bouge vite, car il n'est jamais associé 3 images de suite.

### 4.5 `is_unsure` : faut-il vérifier ce dé ? ([lignes 226-229](../scripts/dice_reader.py#L226-L229))

```python
return die["face_conf"] < min_face_conf or not die["consistent"]
```
Un dé est affiché **en orange** si la lecture est peu sûre **ou** si la face est impossible pour ce type. Avec le lissage, « peu sûre » veut dire que moins de la moitié des dernières images sont d'accord ; sans lissage (`--smooth-frames 1`), cela veut dire que le modèle B est sous 50 % de confiance. `or` et `not` s'écrivent en toutes lettres en Python (`||` et `!` en Java).

### 4.6 `draw_results` : dessiner l'interface ([lignes 232-267](../scripts/dice_reader.py#L232-L267))

**Sur chaque dé** : un cadre, puis l'étiquette sur un fond de couleur pour qu'elle reste lisible.
```python
annotated = frame.copy()                       # on dessine sur une copie, pas sur l'original
cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)        # cadre, épaisseur 2
(text_width, text_height), _ = cv2.getTextSize(label, ...)    # taille du texte en pixels
cv2.rectangle(..., color, -1)                  # épaisseur -1 = rectangle rempli
cv2.putText(annotated, label, ...)             # le texte « D20: 17 »
```
`_` est un nom de variable conventionnel pour une valeur qu'on n'utilise pas.

**Le panneau de droite** :
```python
panel = np.full((annotated.shape[0], PANEL_WIDTH, 3), 30, dtype=np.uint8)
```
- `np.full(forme, 30)` crée une image remplie de la valeur 30, c'est-à-dire gris très foncé, de même hauteur que la photo et de 270 px de large.
- `dtype=np.uint8` : entiers de 0 à 255, le format des images.

Les lignes de texte sont préparées dans une liste de paires `(texte, couleur)` :
```python
for number, die in enumerate(dice, start=1):          # numérotation à partir de 1
    lines.append((f"{number}. {die['type']:>3} = {die['value']:>2}{mark}  ({die['face_conf']:.0%})", color))
total = sum(die["value"] for die in dice)
```
- Dans la **f-string**, `:>3` aligne à droite sur 3 caractères et `:.0%` affiche `0.88` sous la forme `88%`.
- `sum(... for die in dice)` additionne les valeurs de tous les dés : c'est le **total**.

Enfin, on colle la photo et le panneau côte à côte :
```python
return np.hstack([annotated, panel])     # hstack = « horizontal stack »
```
Les deux images doivent avoir la même hauteur. Le résultat fait 640 + 270 = 910 px de large.

### 4.7 `main` : options et boucle caméra ([lignes 270-339](../scripts/dice_reader.py#L270-L339))

**Les options :**

| Option | Défaut | Rôle |
|---|---|---|
| `--source` | `0` | numéro de caméra (`0`, `1`…) ou fichier vidéo |
| `--type-model` | `models/type_best.pt` | modèle A |
| `--cls-model` | `models/cls_best.pt` | modèle B |
| `--conf` | `0.25` | seuil de confiance du modèle A |
| `--imgsz` | `416` | taille d'image du modèle A (celle de son entraînement) |
| `--cls-imgsz` | `128` | taille d'image du modèle B (celle de son entraînement) |
| `--margin` | `0.0` | marge autour du dé avant le modèle B (voir 4.2) |
| `--min-face-conf` | `0.5` | en dessous, la lecture s'affiche en orange |
| `--smooth-frames` | `10` | nombre d'images pour le vote majoritaire (`1` = pas de lissage) |

**Vérifications avant de commencer** : la confiance est entre 0 et 1, les deux fichiers de modèle existent (sinon un message clair), et la vidéo existe si on en donne une.
```python
source = int(args.source) if args.source.isdigit() else args.source
```
`"0"` devient le nombre `0` (une caméra), et `"video.mp4"` reste un texte (un fichier). `a if condition else b` est le ternaire Python (`condition ? a : b`).

**La boucle** : lire une image, la passer aux deux modèles, l'afficher, puis recommencer.
```python
ok, frame = cap.read()                         # ok = False quand la vidéo est finie
dice = read_dice(frame, type_model, number_model, args)
if tracker is not None:
    dice = tracker.update(dice)                # vote majoritaire sur les dernières images
annotated = draw_results(frame, dice, args.min_face_conf)
cv2.imshow(..., annotated)
key = cv2.waitKey(1) & 0xFF                    # touche pressée (attend 1 ms)
```
- **q** ou **ESC** (code 27) arrête le programme.
- **s** enregistre une capture dans `screenshots/dice_<date>.png`, pratique pour la présentation.
- `try: … finally:` garantit que la caméra est **libérée** et les fenêtres fermées, même en cas d'erreur, comme le `finally` de Java.

**Le point d'entrée** : `raise SystemExit(main())` transmet le code de retour de `main()` (0 = OK, 1 = caméra introuvable, 2 = pas de fenêtre possible) au système.

---

## 5. Résultats mesurés

Sur les **375 dés annotés** de `data/raw/set1_dice_number`, avec le pipeline complet :
- **96,3 % des dés lus correctement** (361/375) ;
- **25 ms par image** pour les deux modèles sur CPU, soit environ 40 images par seconde possibles ;
- vérifié à l'œil sur une photo de 5 dés : D20 12, D6 1, D10 0 → **10**, D8 2, D12 12, **total 37**. Tout est juste.

**Limites à dire à l'oral :**
- **96,3 % est optimiste** : ces photos ont très probablement servi à entraîner le modèle B. Pour une mesure honnête, il faut une dizaine de **nouvelles** photos, en comparant le total affiché au vrai total.
- **L'orange ne rattrape pas la plupart des erreurs** : sur les 14 lectures fausses, une seule était en orange. Les 13 autres avaient une confiance ≥ 50 % et une face possible pour le dé.
- **Les erreurs restantes** concernent surtout le D20 (11 lu 13, 20 lu 17, 6 et 9 confondus).

---

## 6. Lancer le programme

Depuis la racine du projet :
```bat
.venv\Scripts\python.exe scripts\dice_reader.py --source 0
```
Utilisez `--source 1` si la mauvaise caméra s'ouvre, ou un fichier vidéo : `--source des.mp4`.

### Sur un autre ordinateur
Il faut **le code + l'environnement Python + les deux modèles** :
1. **Le code** : `git clone https://github.com/ALemasson974/AITrainingDice.git`, ou `git pull` si le dépôt est déjà cloné.
2. **L'environnement** (une seule fois) :
   ```bat
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. **Les deux modèles**, qui ne sont pas dans git : créer le dossier `models\` à la racine du projet et y copier `type_best.pt` et `cls_best.pt`.
4. Lancer la commande ci-dessus.

---

## 7. Questions possibles à l'oral

1. **Pourquoi deux modèles plutôt qu'un ?** Le modèle A trouve et reconnaît les dés ; le modèle B n'a plus qu'à lire un nombre sur une image d'un seul dé, ce qui est une tâche plus simple.
2. **Pourquoi la marge est-elle à 0 ?** Le modèle B doit voir des crops découpés comme à l'entraînement. Mesuré : 96,3 % sans marge, contre 79,5 % avec 25 % de marge.
3. **Pourquoi ne pas forcer le nombre à correspondre au type de dé ?** On l'a mesuré : cela faisait plus d'erreurs, car c'est plus souvent le type (modèle A) qui est faux.
4. **Pourquoi le « 0 » vaut-il 10 ?** C'est la face 10 du D10. Seul le D10 a un 0, donc la règle reste juste même si le type est mal reconnu.
5. **Pourquoi est-ce assez rapide pour du direct ?** Le modèle B travaille sur de petits crops de 128 px, et tous les dés sont lus en un seul appel : 25 ms par image au total.
6. **Comment l'affichage est-il stabilisé ?** Chaque dé est suivi d'image en image (associé au dé le plus proche de l'image précédente), et on affiche le vote majoritaire de ses 10 dernières lectures. Une erreur d'une seule image ne s'affiche donc jamais.
7. **Quel est le prix du lissage ?** Un léger retard : un dé relancé affiche sa nouvelle valeur après environ 5 images (~0,2 s).
