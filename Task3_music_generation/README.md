# AI Music Generation Using LSTM

## Project Overview
This project uses a Long Short-Term Memory (LSTM) neural network to learn note patterns and generate new musical sequences. The generated output is saved as a MIDI file.

## Technologies Used
- Python
- PyTorch
- NumPy
- Pretty MIDI
- Music21

## Features
- Creates starter MIDI examples when no MIDI files are provided.
- Extracts note pitches from MIDI files for training.
- Trains an LSTM model to learn musical note sequences.
- Generates new note sequences using the trained model.
- Saves generated music as a MIDI file.

## Project Structure
- `music_generator.py` — Main program for training and music generation.
- `requirements.txt` — Required Python packages.
- `midi_data/` — MIDI training examples.
- `music_lstm_model.pth` — Saved trained LSTM model.
- `generated_music.mid` — Generated music output.

## Installation
Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the Project
Execute the following command:

```bash
python music_generator.py
```

The program trains the LSTM model and saves the generated music to `generated_music.mid`.

## Dataset Note
If the `midi_data` folder contains no MIDI files, the program automatically creates starter MIDI examples for demonstration and training. You can add your own MIDI files to train on different musical styles.

## Future Improvements
- Train on larger collections of classical or jazz MIDI files.
- Experiment with different neural network architectures.
- Improve note timing, rhythm, and musical structure.
- Add audio rendering and playback features.

## Author
Anshika Patel