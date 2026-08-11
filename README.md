# VOICECLONE-QC V1.0

Application Windows locale pour preparer des datasets de voix, synchroniser les
artefacts avec Google Drive et lancer une conversion RVC via une interface
Gradio.

## Demarrage rapide

1. Installer Python 3.10 ou 3.11. Ces versions sont recommandees pour
   PyTorch, Demucs et les repos RVC.
2. Installer ffmpeg et verifier qu'il est disponible dans le `PATH`.
3. Installer les dependances:

```powershell
.\scripts\setup_windows.bat
```

Si tu utilises Python 3.11:

```powershell
.\scripts\setup_windows.bat -PythonVersion 3.11
```

4. Ajuster `config/settings.json` si le disque ou les ports changent.
5. Lancer l'application:

```powershell
.\scripts\run_app.bat
```

L'interface est exposee sur `0.0.0.0:7860` par defaut, donc accessible depuis
le LAN avec l'adresse IP de ce PC.

## Racine de donnees

La racine runtime par defaut est:

```text
Y:\VOICECLONE
```

Au lancement ou lors d'une operation, l'application verifie que le disque
existe. Si `Y:` est deconnecte, elle leve une erreur explicite au lieu de
continuer dans un mauvais dossier.

La structure creee sous la racine est:

```text
Y:\VOICECLONE\
├── config\
├── dataset_cleaned\
├── models_bank\
├── outputs\
├── staging\
└── temp\
```

## Google Drive

Placer la cle du service account dans:

```text
Y:\VOICECLONE\config\service_account.json
```

Puis renseigner dans `config/settings.json`:

- `google_drive.upload_folder_id`
- `google_drive.return_folder_id` ou laisser `google_drive.return_folder_name`
  a `RVC_Output`

## Compatibilite Colab

La version 1.01 agit comme un pont de fichiers avec Colab. Le ZIP uploade sur
Drive contient directement:

```text
dataset/
├── segment_001.wav
├── segment_002.wav
└── README.txt
```

Le watchdog scanne le dossier Drive `RVC_Output/` toutes les 30 secondes par
defaut. Quand il trouve un couple `.pth` et `.index` dont le nom contient le
nom du comedien, il les rapatrie dans `models_bank/`.

Un fichier `README.txt` est aussi genere automatiquement dans `Y:\VOICECLONE`
pour rappeler le type de notebook Colab a utiliser.

## RVC

La V1 fournit un adaptateur configurable. Renseigner `rvc.command_template`
dans `config/settings.json` pour brancher le repo RVC choisi.

Variables disponibles dans le template:

- `{model_path}`
- `{index_path}`
- `{input_path}`
- `{output_path}`
- `{f0_method}`
- `{device}`
- `{transpose}`
- `{index_rate}`
- `{protect}`

Exemple schematique:

```json
"command_template": "python infer_cli.py --model \"{model_path}\" --index \"{index_path}\" --input \"{input_path}\" --output \"{output_path}\" --f0_method {f0_method}"
```

L'application force `f0_method=rmvpe` par defaut et vide le cache CUDA apres
chaque conversion quand PyTorch est disponible.
