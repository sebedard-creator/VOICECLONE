# VOICECLONE-QC

VOICECLONE-QC is a Windows-first local voice conversion workflow built around
RVC (Retrieval-based Voice Conversion). It prepares voice datasets locally,
uses Google Drive as a bridge to an external Google Colab training notebook,
and runs local RMVPE voice conversion from a browser-based LAN interface.

The project prioritizes reproducible training, audio quality, and keeping
private assets on the local machine.

## What It Does

- Cleans voice datasets with Demucs and creates training-ready audio segments.
- Provides a Split Only mode for source material that is already clean.
- Uploads a Colab-ready `dataset/` ZIP to Google Drive.
- Watches `RVC_Output` on Google Drive and imports matching `.pth` and `.index`
  model pairs into the local model bank.
- Runs local RVC conversion with RMVPE pitch extraction.
- Produces 24-bit / 48 kHz WAV outputs.
- Includes a local audio toolbox for isolation, denoising, and restoration.
- Supports resumable Colab training with checkpoint backups stored on Google
  Drive.
- Provides visual progress feedback and completion alerts for long-running
  local tasks.

## Local Audio Toolbox

The **Non-Colab Features** page contains independent local processors. These
tools do not require a Google Colab training session and do not modify the RVC
model bank.

### Dataset Preparation

- **Demucs dataset cleaning:** isolates vocals before automatic slicing,
  normalization, and dataset export.
- **Split Only:** skips Demucs while retaining the automatic slicing and export
  workflow for sources that are already clean.

### Standalone Processing

- **Demucs vocal isolation:** produces a voice-focused file without slicing.
  Its quality control trades processing time for additional separation
  passes.
- **DeepFilterNet:** applies adjustable-strength speech denoising for a faster,
  alternative cleanup pass.
- **UVR:** offers an alternative vocal-isolation pipeline and selectable source
  separation models, with a separate quality/speed control.
- **Resemble Enhance:** performs high-fidelity generative audio restoration.
  Solver selection and reconstruction quality allow slower, more careful
  restoration when required.
- **VoiceFixer:** restores degraded vocal recordings through selectable
  restoration modes.

Each processor writes a separately named WAV output, keeps the original input
unchanged, and shows an active processing indicator while the task runs.

## Local Workflow Controls

For controlled auditions, `scripts/compare_rvc.py --model <model.pth> --guide <guide.wav>`
creates five conversions of the same guide, changing only the index influence or
consonant protection relative to the baseline. An optional `--reference <voice.wav>`
adds a timbre reference. Outputs are placed in `outputs/comparisons/` with an
`ecouter.html` listening page, original WAV files, and a parameter/measurement log.
Listening copies use matched integrated loudness through constant gain, without
compression or limiting; original renders remain available. A fixed random seed
reduces run-to-run variation. Like other outputs, these auditions are removed by
the application's explicit cache-clearing operation.

In the installed RVC runtime, lower Protect values preserve more of the original
unvoiced features during index blending; 0.5 disables this protection. The interface
help now reflects the implementation rather than the inverted upstream help text.

- A single-task queue prevents multiple heavy audio operations from competing
  for the same machine resources.
- The RVC conversion controls expose pitch transposition, index influence, and
  consonant/breath protection, while RMVPE remains the selected F0 method.
- The local model bank can be refreshed from the interface after Drive imports.
- The Colab notebook URL can be configured from the interface with either a
  Google Drive notebook file link or a direct Colab link.
- Transient folders can be cleared without deleting trained models, credentials,
  runtimes, or application settings.

## Architecture

```text
Local audio files
  -> VOICECLONE-QC preprocessing
  -> Google Drive dataset ZIP
  -> Google Colab RVC training
  -> Google Drive RVC_Output
  -> Local models_bank
  -> Local RVC conversion
```

## Requirements

- Windows 10 or 11
- Python 3.11
- FFmpeg available on `PATH`
- Google Drive access and an OAuth client configured for the application
- An NVIDIA GPU is optional for local conversion; the application also supports
  CPU execution
- A Google Colab runtime for RVC model training

## Setup

1. Clone the repository into the intended local project directory.
2. Create the Python environment using the setup script in `scripts/`.
3. Copy `config/settings.example.json` to `config/settings.json` and configure
   local paths and Google Drive OAuth settings.
4. Keep OAuth credentials, tokens, trained models, audio files, and runtime
   downloads outside Git. The provided `.gitignore` is designed for this.
5. Start the application with `scripts/run_app.bat`.
6. Open the local web interface at `http://localhost:7860/`.

## Colab Workflow

### Drive storage and local archives (bridge 1.3.0)

The **Cloud > Espace Drive et archives locales** panel provides this workflow:

1. Finish training, retrieve the final model, and stop the Colab runtime.
2. Click **Archiver Drive localement**. The three configured VOICECLONE folders
   are downloaded into `archives_drive/` on the project disk. Identical content
   shares one stored copy. Downloads and the complete snapshot are verified;
   interrupted downloads can be retried without downloading verified objects again.
3. Select that snapshot, confirm that Colab is stopped, and click **Liberer Drive**.
   This permanently removes only the unchanged files in that verified snapshot.
   It does not empty the Google Drive trash or delete folders. New or changed
   files block cleanup. Local models and archives remain available.
4. To continue training later, select the snapshot and the actor name, then click
   **Remettre cette voix sur Drive**. Wait for completion before running Colab
   with `RUN_MODE=resume` and a larger total epoch count. Resume restores prepared
   training data and checkpoints; it does not need to upload the dataset ZIP again.

The local archive uses `objects/` for content and `snapshots/` for inventories.
Keep both directories together and include `archives_drive/` in your disk backups.
It is excluded from Git and from the application's cache-clearing operation.
Archiving alone never removes Drive files. Cleanup is an explicit separate action
because it includes all three project folders, possibly containing several voices.

Identical dataset uploads are reused. Uploading different content under an existing
name is refused until the old data is archived and removed, preventing silent
replacement and accumulation of same-name datasets.

Bridge 1.3.0 keeps prepared data once and one verified, matching G/D checkpoint
pair. A new pair is uploaded and verified before it replaces the previous pair;
the temporary handover still needs space for both pairs. The steady-state duplicate
weights previously kept in `experiment/` are eliminated. Resume also understands
v1.2.x backups and chooses the newest complete pair rather than an outdated
experiment copy. New runs refuse to overwrite an existing training backup.

The archive/checkpoint behavior is covered by `python -m unittest discover -s tests -v`.
Regenerate the notebook from the maintained helper with
`python scripts/update_colab_notebook.py` after changing `colab/checkpoint_store.py`.

The included `VOICECLONE_QC_RVC_Bridge.ipynb` is a reproducible training bridge.
Run the Google Drive mount cell immediately after its configuration cell to
authorize Drive before the longer dependency and model-download steps begin.

For a new model, run the notebook from top to bottom. For a stopped training
session, set `RUN_MODE` to `resume`; the notebook restores the experiment and
checkpoint backup from Google Drive before continuing toward the configured
total epoch count.

## Security and Privacy

Do not commit the contents of `config/settings.json`, OAuth client files,
OAuth tokens, service-account credentials, trained models, audio datasets, or
generated outputs. These are intentionally ignored by Git.

The Gradio interface can be exposed on a LAN. Only grant access to trusted
users and networks. Use voice data and trained models only with the necessary
permission from the voice owner.

## Project Notes

This is a practical local workflow rather than a hosted service. Training is
performed externally in Google Colab; VOICECLONE-QC does not automate or
control the Colab user interface.

The code and user interface are currently in French.

Conçu par Sébastien Bédard
