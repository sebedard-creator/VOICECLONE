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
- Includes optional local, non-Colab audio tools for isolation, denoising, and
  restoration.
- Supports resumable Colab training with checkpoint backups stored on Google
  Drive.

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
