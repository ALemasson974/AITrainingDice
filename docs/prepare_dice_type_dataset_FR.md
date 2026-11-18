# `prepare_dice_type_dataset.py` : explication détaillée

> Explication du script [`scripts/prepare_dice_type_dataset.py`](../scripts/prepare_dice_type_dataset.py)
> dans sa version du dataset v2 (set1 + set2). Les numéros de ligne correspondent
> à cette version : s'ils changent après une modification du script, mettez-les à jour ici.

## 1. À quoi sert ce script ?

L'export de CVAT ne peut pas être utilisé tel quel pour entraîner YOLO, pour deux raisons :

1. **Les images et les labels ne sont pas rangés comme YOLO l'attend.** CVAT met les photos dans `captures/` et les labels dans `labels/train/`. Or YOLO retrouve le label d'une image en remplaçant `/images/` par `/labels/` dans son chemin. Il faut donc des dossiers `images/` et `labels/` parallèles.
2. **Il n'y a pas d'ensemble de validation.** Tout est dans `train`, alors qu'on a besoin d'images que le modèle ne voit jamais pendant l'entraînement pour mesurer ses vraies performances.

Le script **lit** les exports bruts sans jamais les modifier, puis **construit** un dataset propre :

```text
AVANT (données brutes, une par séance photo)    APRÈS (dataset généré)
data/raw/                                       datasets/dice_type/
├── set1_dice_type/                             ├── data.yaml          ← config pour YOLO
│   ├── data.yaml        (noms des classes)     ├── split_report.txt   ← rapport du découpage
│   ├── captures/*.jpg   (photos)               ├── images/
│   └── labels/train/*.txt (annotations)        │   ├── train/*.jpg    (77 photos)
└── set2_dice_type/                             │   └── val/*.jpg      (19 photos)
    └── (même structure)                        └── labels/
                                                    ├── train/*.txt
                                                    └── val/*.txt
```

**Rappel du format d'un label YOLO.** Chaque fichier `.txt` a le même nom que sa photo et contient une ligne par dé :
```
1 0.4523 0.6120 0.0910 0.1185
│   │      │      │      └─ hauteur de la boîte
│   │      │      └──────── largeur de la boîte
│   │      └─────────────── y du centre
│   └────────────────────── x du centre
└────────────────────────── numéro de classe (1 = D8)
```
Les quatre valeurs sont **normalisées entre 0 et 1**, c'est-à-dire divisées par la largeur ou la hauteur de l'image. Le label reste donc valable quelle que soit la taille de l'image.

---

## 2. Les imports ([lignes 39-45](../scripts/prepare_dice_type_dataset.py#L39-L45))

```python
import argparse                 # lire les options de la ligne de commande (--source, --seed…)
import random                   # mélange aléatoire (reproductible grâce à la graine)
import shutil                   # copier et supprimer des fichiers/dossiers
from datetime import datetime   # manipuler des dates/heures
from pathlib import Path        # manipuler des chemins de fichiers
import yaml                     # lire et écrire les fichiers .yaml (bibliothèque PyYAML)
```
`import x` correspond à peu près à l'`import` de Java. `from x import y` importe directement `y`, qu'on peut ensuite écrire sans le préfixe `x.`.

**`pathlib.Path`** est utilisé partout. L'opérateur `/` y sert à assembler des chemins :
```python
source / "labels" / "train"   # → data/raw/set1_dice_type/labels/train
```
Cela fonctionne sous Windows comme sous Linux, sans gérer soi-même les `\` ou `/`.

---

## 3. Les trois fonctions utilitaires

### 3.1 `read_capture_time` : lire l'heure dans le nom du fichier ([lignes 48-58](../scripts/prepare_dice_type_dataset.py#L48-L58))

```python
def read_capture_time(image_path):
    time_text = image_path.stem.split("_", 1)[1]
    return datetime.strptime(time_text, "%Y%m%d_%H%M%S_%f")
```
Voici ce qui se passe pour `frame_20261118_174653_953800.jpg` :

| Étape | Résultat |
|---|---|
| `image_path.stem` (nom sans extension) | `"frame_20261118_174653_953800"` |
| `.split("_", 1)` (coupe au **premier** `_` seulement) | `["frame", "20261118_174653_953800"]` |
| `[1]` (2ᵉ élément, les indices commencent à 0 comme en Java) | `"20261118_174653_953800"` |
| `datetime.strptime(..., "%Y%m%d_%H%M%S_%f")` | date du 18/11/2026 à 17:46:53.953800 |

`strptime` signifie *string parse time*. Le format décrit le texte : `%Y` = année, `%m` = mois, `%d` = jour, `%H%M%S` = heure, minute, seconde, `%f` = microsecondes. C'est l'équivalent de `LocalDateTime.parse(texte, DateTimeFormatter.ofPattern(...))` en Java.

**Pourquoi le nom du fichier et pas la date de modification ?** La date de modification change quand on copie le fichier, par exemple lors de l'export CVAT ou d'une copie sur clé USB. Le nom, lui, est fixé par `webcam_capture.py` au moment de la photo.

⚠️ **Conséquence :** une photo nommée autrement, comme `IMG_1234.jpg` venant d'un téléphone, fait planter le script.

### 3.2 `group_into_scenes` : regrouper les photos en « scènes » ([lignes 61-82](../scripts/prepare_dice_type_dataset.py#L61-L82))

**Le problème qu'elle résout :** quand on prend des photos à quelques secondes d'intervalle, elles sont quasiment identiques. Si l'une va dans `train` et sa « jumelle » dans `val`, le modèle est évalué sur une image qu'il a pratiquement déjà vue. Les scores paraissent alors meilleurs qu'ils ne le sont : c'est une **fuite de données** (*data leakage*).

**La solution :** on regroupe les photos proches dans le temps en **scènes**, et une scène entière va soit dans train, soit dans val.

```python
sorted_paths = sorted(image_paths, key=read_capture_time)
```
Cette ligne trie les photos par heure de prise. `key=read_capture_time` dit à Python de comparer les photos selon le résultat de cette fonction ; c'est comme passer un `Comparator.comparing(...)` en Java. On passe la fonction elle-même, sans parenthèses, et non son résultat.

```python
scenes = []
current_scene = []
previous_time = None                      # None ≈ null en Java
for path in sorted_paths:
    time = read_capture_time(path)
    if previous_time is not None:
        seconds_since_previous = (time - previous_time).total_seconds()
        if seconds_since_previous > gap_seconds:   # plus de 5 s → nouvelle scène
            scenes.append(current_scene)            # on range la scène terminée
            current_scene = []                      # et on en commence une nouvelle
    current_scene.append(path)
    previous_time = time
if current_scene:                         # ne pas oublier la dernière scène
    scenes.append(current_scene)
```
- Soustraire deux `datetime` donne une durée (`timedelta`), et `.total_seconds()` la convertit en nombre de secondes.
- `if current_scene:` est vrai si la liste n'est pas vide. En Python, une liste vide vaut `False` dans un `if`.
- Le résultat est une **liste de listes** (comme `List<List<Path>>`) : chaque scène est une liste de photos.

**Exemple réel** (scène 21 du set 2) :
```
17:48:14 → 17:48:17 → 17:48:21 → 17:48:24 → 17:48:26   écarts ≤ 5 s → une scène de 5 photos
17:48:42                                                écart de 16 s → nouvelle scène
```
L'écart est mesuré par rapport à la photo **précédente**, pas à la première de la scène. Une rafale régulière peut donc durer 30 secondes et rester une seule scène : c'est le cas de la scène 8 du set 1, avec 12 photos sur 31 s. C'est voulu, car les dés n'ont presque pas bougé pendant la rafale.

### 3.3 `choose_validation_scenes` : choisir les scènes de validation ([lignes 85-107](../scripts/prepare_dice_type_dataset.py#L85-L107))

```python
total_images = sum(len(scene) for scene in scenes)          # 96
target_val_images = round(total_images * val_fraction)       # round(96 × 0.2) = round(19.2) = 19
```
`sum(len(scene) for scene in scenes)` est une **expression génératrice** : elle additionne la taille de chaque scène. C'est l'équivalent de `scenes.stream().mapToInt(List::size).sum()` en Java.

```python
shuffled_indices = list(range(len(scenes)))     # [0, 1, 2, …, 37]
random.Random(seed).shuffle(shuffled_indices)   # mélange, par ex. [14, 3, 29, 0, …]
```
- On mélange les **numéros** des scènes, pas les scènes elles-mêmes, pour garder la liste d'origine intacte.
- `random.Random(seed)` crée un générateur aléatoire avec une **graine** fixe (42). La même graine donne toujours le même mélange. Le découpage est donc **aléatoire mais reproductible** : relancer le script redonne exactement le même train/val, ce qui est indispensable pour comparer deux entraînements honnêtement. C'est l'équivalent de `new Random(42)` en Java.

```python
val_scene_indices = set()        # un set = ensemble sans doublons (≈ HashSet)
val_image_count = 0
for index in shuffled_indices:
    scene_size = len(scenes[index])
    if val_image_count + scene_size <= target_val_images:   # la scène tient encore ?
        val_scene_indices.add(index)
        val_image_count += scene_size
    if val_image_count == target_val_images:                # objectif atteint → stop
        break
return val_scene_indices
```
On parcourt les scènes dans l'ordre mélangé et on prend chaque scène qui **ne fait pas dépasser** l'objectif. C'est un algorithme **glouton** (*greedy*) : il prend ce qui rentre, sans revenir en arrière.

Par exemple, avec 15 photos déjà en val et un objectif de 19 :
- une scène de 6 photos donnerait 21, elle est **sautée** et reste en train ;
- une scène de 4 photos donne exactement 19, elle est **prise**, et on s'arrête.

On ne retombe donc pas toujours exactement sur 20 % : on prend le total le plus proche possible sans dépasser. Pour le dataset v2, on tombe pile sur 19.

---

## 4. La fonction `main()` : le programme principal ([lignes 110-241](../scripts/prepare_dice_type_dataset.py#L110-L241))

### Les options ([lignes 111-127](../scripts/prepare_dice_type_dataset.py#L111-L127))

| Option | Défaut | Rôle |
|---|---|---|
| `--source` | `set1_dice_type` et `set2_dice_type` | un ou plusieurs dossiers bruts. `nargs="+"` signifie « une ou plusieurs valeurs » et donne une liste Python |
| `--output` | `datasets/dice_type` | dossier où écrire le dataset |
| `--val-fraction` | `0.2` | 20 % en validation : assez pour mesurer, tout en gardant 80 % de nos rares données pour apprendre |
| `--gap-seconds` | `5.0` | écart au-delà duquel une nouvelle scène commence |
| `--seed` | `42` | graine du mélange (reproductibilité) |

`argparse` transforme `--val-fraction` en `args.val_fraction` : le tiret devient un underscore, car un tiret n'est pas autorisé dans un nom de variable Python.

### Étape 1 : lire et vérifier les classes ([lignes 142-157](../scripts/prepare_dice_type_dataset.py#L142-L157))

```python
with open(source / "data.yaml", encoding="utf-8") as file:
    source_config = yaml.safe_load(file)
```
- `with … as file:` ouvre le fichier et le **ferme automatiquement** à la fin du bloc, même en cas d'erreur. C'est l'équivalent du *try-with-resources* de Java.
- `yaml.safe_load` transforme le texte YAML en objets Python ; ici un dictionnaire `{"names": {0: "D6", 1: "D8", 2: "D10", 3: "D12", 4: "D20"}, ...}`. Un dictionnaire (`dict`) correspond à une `HashMap`.

```python
if class_names is None:
    class_names = source_config["names"]          # 1ʳᵉ source : on mémorise l'ordre
elif source_config["names"] != class_names:       # sources suivantes : on compare
    raise SystemExit("ERROR: ...")                # arrêt immédiat avec un message
```
**Pourquoi cette vérification ?** Un label contient un **numéro** de classe, pas un nom. Si le set 2 avait été exporté avec un autre ordre (par exemple `0: D8, 1: D6`), le chiffre `1` voudrait dire D6 dans un set et D8 dans l'autre. Le modèle apprendrait des étiquettes fausses **sans aucun message d'erreur**. Mieux vaut donc arrêter le script tout de suite. `raise SystemExit(...)` correspond à peu près à `System.exit()` avec un message.

`elif` est la contraction de `else if`.

### Étape 2 : associer chaque image à son label ([lignes 159-185](../scripts/prepare_dice_type_dataset.py#L159-L185))

```python
source_images = {path.stem: path for path in image_folder.glob("*.jpg")}
source_labels = {path.stem: path for path in label_folder.glob("*.txt")}
```
- `glob("*.jpg")` liste tous les fichiers `.jpg` du dossier.
- `{clé: valeur for … in …}` est une **compréhension de dictionnaire** : elle construit un dictionnaire en une ligne, ici `{"frame_2026…": chemin_complet, ...}`. En Java, on ferait une boucle `for` avec `map.put(...)`.
- La clé est le **stem**, c'est-à-dire le nom sans extension. Une image et son label vont ensemble quand ils ont le même stem : `frame_X.jpg` ↔ `frame_X.txt`.

```python
paired_stems         = sorted(set(source_images) & set(source_labels))   # dans les deux
images_without_label = sorted(set(source_images) - set(source_labels))   # image seule
labels_without_image = sorted(set(source_labels) - set(source_images))   # label seul
```
Ce sont des **opérations d'ensembles**, comme en maths : `&` = intersection, `-` = différence. `set(dictionnaire)` donne l'ensemble de ses clés.

Les fichiers orphelins sont **ignorés**, avec un message `SKIPPED`. Une image sans label serait comprise par YOLO comme « il n'y a aucun dé sur cette photo », ce qui est faux ici et apprendrait au modèle à ne rien détecter. C'est le cas de `frame_20261117_184714_170925.jpg` dans le set 1.

```python
for stem in paired_stems:
    if stem in images_by_stem:          # déjà vu dans une source précédente ?
        raise SystemExit("ERROR: ... labelled in two sources ...")
    images_by_stem[stem] = source_images[stem]
    labels_by_stem[stem] = source_labels[stem]
```
On ajoute les paires de cette source aux dictionnaires **globaux**, qui regroupent toutes les sources. Deuxième vérification : une même photo annotée dans deux sets compterait deux fois, et en plus les deux copies s'écraseraient dans le dossier de sortie. Le script s'arrête donc là aussi.

### Étape 3 : scènes et choix de la validation ([lignes 189-194](../scripts/prepare_dice_type_dataset.py#L189-L194))

```python
paired_images = list(images_by_stem.values())
scenes = group_into_scenes(paired_images, args.gap_seconds)
val_scene_indices = choose_validation_scenes(scenes, args.val_fraction, args.seed)
```
On applique les deux fonctions vues plus haut à **toutes** les photos de toutes les sources réunies. Les séances photo sont espacées de plusieurs heures, donc elles tombent toujours dans des scènes différentes.

### Étape 4 : reconstruire les dossiers et copier ([lignes 196-218](../scripts/prepare_dice_type_dataset.py#L196-L218))

```python
for sub_folder in ("images", "labels"):
    if (output / sub_folder).exists():
        shutil.rmtree(output / sub_folder)     # supprime le dossier et tout son contenu
```
On efface l'ancien découpage avant d'écrire le nouveau. Sinon, une photo qui passe de val à train entre deux exécutions se retrouverait **dans les deux dossiers**, ce qui recréerait une fuite de données. Seuls `images/` et `labels/` sont supprimés, jamais les données brutes.

```python
for scene_number, scene in enumerate(scenes):
    split = "val" if scene_number in val_scene_indices else "train"
```
- `enumerate(...)` donne à la fois le numéro et l'élément : `(0, scène0), (1, scène1), …`. Cela remplace le `for (int i = 0; …)` de Java.
- `a if condition else b` est l'**opérateur ternaire** de Python, l'équivalent de `condition ? a : b` en Java.

```python
start = read_capture_time(scene[0]).strftime("%H:%M:%S")    # 1ʳᵉ photo de la scène
end   = read_capture_time(scene[-1]).strftime("%H:%M:%S")   # dernière photo
report_lines.append(f"scene {scene_number:2d}  {start}-{end}  {len(scene):2d} images  -> {split}")
```
- `scene[-1]` désigne le **dernier** élément. En Python, les indices négatifs partent de la fin.
- `strftime` fait l'inverse de `strptime` : il transforme une date en texte.
- `f"..."` est une **f-string** : ce qui est entre `{}` est remplacé par sa valeur. `:2d` veut dire « entier sur 2 caractères », pour aligner les colonnes du rapport.

```python
for image_path in scene:
    label_path = labels_by_stem[image_path.stem]
    (output / "images" / split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output / "images" / split / image_path.name)
    shutil.copy2(label_path, output / "labels" / split / label_path.name)
    count_per_split[split] += 1
```
- `mkdir(parents=True, exist_ok=True)` crée le dossier et ses parents si besoin, sans erreur s'il existe déjà. C'est l'équivalent de `mkdir -p`.
- `shutil.copy2` **copie** le fichier, en conservant ses dates. Les originaux restent intacts dans `data/raw/`.
- `parents=True` est un **argument nommé** : on précise le nom du paramètre à l'appel. Cela n'existe pas en Java.

### Étape 5 : écrire le `data.yaml` pour YOLO ([lignes 220-230](../scripts/prepare_dice_type_dataset.py#L220-L230))

```python
dataset_config = {"train": "images/train", "val": "images/val", "names": class_names}
yaml.safe_dump(dataset_config, file, sort_keys=False)
```
Résultat :
```yaml
train: images/train
val: images/val
names:
  0: D6
  1: D8
  2: D10
  3: D12
  4: D20
```
C'est le fichier qu'on donne à l'entraînement avec `--data`. **Il n'y a volontairement pas de clé `path:`.** Ultralytics prend alors comme racine le dossier qui contient `data.yaml`, et le dataset fonctionne sur l'ordinateur de n'importe quel membre du groupe, quel que soit le dossier d'où on lance la commande. `sort_keys=False` garde l'ordre d'écriture des clés.

### Étape 6 : le rapport ([lignes 232-241](../scripts/prepare_dice_type_dataset.py#L232-L241))

Le script écrit `split_report.txt` et affiche dans le terminal la liste des scènes avec leur destination, puis le résumé :
```
sources: data/raw/set1_dice_type, data/raw/set2_dice_type
train: 77 images | val: 19 images | 38 scenes (gap > 5.0s) | seed 42
```
Ce rapport sert de **preuve** pour `CHANGES.md` et la présentation : il montre exactement avec quelles données chaque modèle a été entraîné.

### Le point d'entrée ([lignes 244-245](../scripts/prepare_dice_type_dataset.py#L244-L245))

```python
if __name__ == "__main__":
    main()
```
Ce bloc s'exécute seulement quand on **lance** le fichier directement (`python prepare_dice_type_dataset.py`), et pas quand un autre script l'**importe**. C'est l'équivalent du `public static void main` de Java.

---

## 5. Comment le lancer

Depuis la racine du projet :
```bat
:: Dataset v2 (set1 + set2), utilisé pour le run 2
.venv\Scripts\python.exe scripts\prepare_dice_type_dataset.py

:: Reconstruire exactement le dataset v1 du run 1
.venv\Scripts\python.exe scripts\prepare_dice_type_dataset.py --source data/raw/set1_dice_type

:: Voir toutes les options
.venv\Scripts\python.exe scripts\prepare_dice_type_dataset.py --help
```
**Ce qui doit s'afficher :** le nombre de paires par source, les éventuels `SKIPPED`, la liste des scènes, puis `train: 77 images | val: 19 images`. Si ça ne va pas, vous verrez soit un message `ERROR:` (ordre des classes différent ou photo en double), soit une erreur `ValueError` sur une date, ce qui signifie qu'une photo n'est pas nommée `frame_…`.

---

## 6. Limites à connaître

- **Le seuil de 5 s est une approximation.** Deux photos prises à 10 s d'intervalle sans bouger les dés forment deux scènes, mais restent presque identiques.
- **Le découpage n'équilibre pas les classes.** Ce n'est pas un problème ici, car chaque photo contient les 5 dés : chaque classe a exactement 19 boîtes en validation.
- **Ajouter des photos change le découpage.** Le mélange porte sur une liste de scènes différente. Pour comparer deux modèles, il faut les évaluer sur le **même** ensemble de validation, d'où le `run1_on_v2val`.
- **19 images de validation, c'est peu.** Une seule détection en plus ou en moins fait bouger les scores, donc il faut rester prudent sur les petites différences.

---

## 7. Questions possibles à l'oral

1. **Pourquoi ne pas tirer 20 % des photos au hasard ?** Des photos quasi identiques se retrouveraient dans train et dans val (fuite de données), et les scores de validation seraient trop optimistes.
2. **À quoi sert la graine `seed=42` ?** Elle rend le mélange reproductible : mêmes données + même graine = même découpage, ce qui permet de comparer les entraînements.
3. **Pourquoi vérifier l'ordre des classes entre les sets ?** Les labels contiennent des numéros, pas des noms. Un ordre différent étiquetterait silencieusement les dés de travers.
4. **Pourquoi ignorer une image sans label ?** YOLO la prendrait pour une image sans aucun dé et apprendrait à ne rien détecter.
5. **Pourquoi supprimer `images/` et `labels/` avant de recopier ?** Pour qu'une photo passée de val à train ne reste pas aussi dans l'ancien dossier val.
