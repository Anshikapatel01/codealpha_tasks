import os
import glob
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configuration
DATA_DIR = "midi_data"
OUTPUT_FILE = "generated_music.mid"
SEQUENCE_LENGTH = 32
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(DATA_DIR, exist_ok=True)


def create_sample_midi(path, seed=0):
    """Create a small original MIDI melody for initial training."""
    rng = np.random.default_rng(seed)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    notes = [60, 62, 64, 67, 69, 67, 64, 62]

    time = 0.0
    for _ in range(8):
        for note in notes:
            pitch = int(np.clip(note + rng.choice([-2, 0, 0, 0, 2]), 48, 84))
            duration = float(rng.choice([0.25, 0.5]))
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=90,
                    pitch=pitch,
                    start=time,
                    end=time + duration
                )
            )
            time += duration

    midi.instruments.append(instrument)
    midi.write(path)


def load_midi_notes():
    """Read MIDI files and extract note pitches."""
    sequences = []

    files = glob.glob(os.path.join(DATA_DIR, "*.mid"))
    if not files:
        print("No MIDI dataset found. Creating starter MIDI examples...")
        for i in range(8):
            create_sample_midi(os.path.join(DATA_DIR, f"sample_{i}.mid"), i)
        files = glob.glob(os.path.join(DATA_DIR, "*.mid"))

    for file_path in files:
        try:
            midi = pretty_midi.PrettyMIDI(file_path)
            notes = [
                note.pitch
                for instrument in midi.instruments
                for note in sorted(instrument.notes, key=lambda n: n.start)
            ]
            if len(notes) > SEQUENCE_LENGTH:
                sequences.extend(
                    [
                        notes[i:i + SEQUENCE_LENGTH + 1]
                        for i in range(len(notes) - SEQUENCE_LENGTH)
                    ]
                )
        except Exception as error:
            print(f"Skipping {file_path}: {error}")

    if not sequences:
        raise ValueError("Not enough MIDI notes to train. Add longer .mid files.")

    return np.array(sequences, dtype=np.int64)


class MusicLSTM(nn.Module):
    """LSTM model that learns note sequences."""

    def __init__(self, vocab_size=128, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


def train_model(sequences):
    inputs = torch.tensor(sequences[:, :-1], dtype=torch.long)
    targets = torch.tensor(sequences[:, -1], dtype=torch.long)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MusicLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_function = nn.CrossEntropyLoss()

    print(f"Training LSTM on {DEVICE}...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "music_lstm_model.pth")
    return model


def generate_music(model, seed_notes, number_of_notes=128):
    """Generate new note pitches from the trained model."""
    model.eval()
    generated = list(seed_notes)

    with torch.no_grad():
        for _ in range(number_of_notes):
            context = generated[-SEQUENCE_LENGTH:]
            if len(context) < SEQUENCE_LENGTH:
                context = [60] * (SEQUENCE_LENGTH - len(context)) + context

            x = torch.tensor([context], dtype=torch.long).to(DEVICE)
            probabilities = torch.softmax(model(x)[0], dim=0)
            next_note = torch.multinomial(probabilities, 1).item()
            generated.append(next_note)

    return generated


def save_midi(notes, output_path):
    """Convert generated notes into a playable MIDI file."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    time = 0.0

    for pitch in notes:
        duration = 0.4
        instrument.notes.append(
            pretty_midi.Note(
                velocity=85,
                pitch=int(np.clip(pitch, 48, 84)),
                start=time,
                end=time + duration
            )
        )
        time += duration

    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"Generated music saved to: {output_path}")


def main():
    sequences = load_midi_notes()
    print(f"Training examples: {len(sequences)}")

    model = train_model(sequences)
    seed = sequences[0, :-1].tolist()
    generated_notes = generate_music(model, seed)
    save_midi(generated_notes, OUTPUT_FILE)

    print("Done! Open generated_music.mid in a MIDI player to listen.")


if __name__ == "__main__":
    main()