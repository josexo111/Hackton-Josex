# ⚡ GEN-GLITCH
### Bio-Hacker DNA → Audio/Visual Transduction Engine

> *"What if your genome sounded like a glitch album?"*

---

## What is this

GEN-GLITCH takes a DNA sequence in `.fasta` format and turns it into two things: **sound** and **real-time ASCII art in your terminal**. The idea is pretty simple — DNA carries information, information has entropy, and entropy can be heard.

Each nucleotide base (A, T, C, G) gets a musical frequency. Each k-mer window generates a "glitch state" based on its complexity. When the DNA gets too chaotic (entropy > 1.9 bits), everything breaks: sawtooth waves, inverted colors, flashing banners. CRITICAL GLITCH.

---

## How it works

The pipeline chains 4 modules together:

```
[FASTA] → Bio-Kernel → Chaos Engine → Sonic Synthesis + Glitch Visualizer
                                              ↓                  ↓
                                           .wav file       Terminal waterfall
```

**Bio-Kernel** (`bio_kernel.py`) — reads the FASTA, computes Shannon entropy across sliding windows, detects ORFs (open reading frames), and flags mutation hotspots where the local GC% drifts away from the sequence baseline.

**Chaos Engine** (`chaos_engine.py`) — takes the k-mer windows and converts them into `ChaosVectors`. Each vector packs audio parameters (frequency, waveform type, FM intensity) and visual parameters (density, jitter, color inversion). The state machine has 4 levels: SILENT → STABLE → TURBULENT → CRITICAL.

**Sonic Synthesis** (`sonic_synthesis.py`) — actual additive synthesis. Square and sawtooth waves are built from Fourier series to avoid aliasing. Notes use Hanning windows so there are no clicks between transitions. Active codons trigger Bio-Chords (all three bases playing at once).

**Glitch Visualizer** (`glitch_visualizer.py`) — ASCII waterfall in the terminal, frame-locked to audio at 12.5 fps (80ms per note). Glyphs scale from ` ` to `@` with entropy density. In CRITICAL state, individual characters get 256-color assignments via a deterministic hash — reproducible glitch, not random noise.

---

## Installation

```bash
git clone https://github.com/your-username/gen-glitch
cd gen-glitch
pip install numpy
```

That's it. Everything else is Python stdlib.

---

## Usage

```bash
# Basic — processes the first FASTA record
python gen_glitch.py --fasta my_sequence.fasta

# With options
python gen_glitch.py \
  --fasta genome.fasta \
  --output output.wav \
  --kmer 100 \
  --step 25 \
  --min-orf 150 \
  --max-records 3

# Visual only, no audio
python gen_glitch.py --fasta sequence.fasta --no-audio

# Audio only, no terminal
python gen_glitch.py --fasta sequence.fasta --no-visual
```

### Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--fasta` | required | Input `.fasta` file |
| `--output` | `<name>.wav` | Output `.wav` file |
| `--kmer` | 100 | Sliding window size (bp) |
| `--step` | 25 | Step between windows (75% overlap) |
| `--min-orf` | 150 | Minimum ORF length to detect |
| `--max-records` | 1 | How many FASTA records to process |

---

## The sound system

The 4 bases map to a Dm7 chord — chosen because the minor seventh has that biological tension to it:

| Base | Note | Hz |
|------|------|----|
| A | A4 | 440.00 |
| T | D4 | 293.66 |
| C | C5 | 523.25 |
| G | G4 | 392.00 |

Waveform type changes with glitch state:
- **SILENT** → sine (near silence)
- **STABLE** → clean sine
- **TURBULENT** → square wave (odd harmonics, feels electric)
- **CRITICAL** → sawtooth (all harmonics, maximum chaos)

When an active codon is detected and chord_weight is high enough, the three bases synthesize simultaneously as a Bio-Chord.

---

## The visual system

Shannon entropy maps to glyph density:

```
0.0 – 0.2  →   (void)
0.2 – 0.5  →  · :
0.5 – 0.75 →  ~ * +
0.75 – 1.0 →  # $ @
```

Base colors in terminal:
- **A** → matrix green `\033[92m`
- **T** → amber yellow
- **C** → electric cyan
- **G** → magenta

In CRITICAL state: the entire row inverts with `ANSI 7m`, red background, and a blinking banner fires with the glitch position and entropy value.

Active ORFs are marked with `[0]` `[1]` `[2]` depending on the reading frame.

---

## Sample output

```
[GEN-GLITCH] Initiating transduction: ecoli_k12.fasta
[GEN-GLITCH] k-mer: 100bp | step: 25bp | min ORF: 150bp

  Sequence ID:   NC_000913.3
  Length:        4,641,652 bp
  Global GC:     50.79%
  ORFs detected: 847
  Hotspots:      203
  Max H(s):      1.998412
  Mean H(s):     1.743291

     POS   BASE    STATE       H(s)    GC-SKEW   WAVEFORM    WATERFALL SIGNAL
────────────────────────────────────────────────────────────────────────────────
      25    A      STABLE      1.7821  +0.032    sine        A~~~~~*~~~~~[0]
      50    T      TURBULENT   1.8943  -0.011    square      T######+####
      75    G      CRITICAL!   1.9934  +0.201    sawtooth    ⚡ CRITICAL GLITCH @ pos 75 ...
```

---

## Project structure

```
gen-glitch/
├── gen_glitch.py          # Entry point and main pipeline
├── bio_kernel.py          # Module A: FASTA parsing, entropy, ORFs
├── chaos_engine.py        # Module B: state machine, ChaosVectors
├── sonic_synthesis.py     # Module C: DSP audio synthesis
├── glitch_visualizer.py   # Module D: ANSI terminal waterfall
└── codon_map.json         # Sonic and visual profiles per amino acid
```

---

## Stack

- **Python 3.11+**
- **NumPy** — audio synthesis and signal ops
- **wave** (stdlib) — WAV export, no external audio libs needed
- **math / collections / dataclasses** — everything else

No Flask. No Electron. No weird frameworks. Just Python and a terminal.

---

## Credits

Final programming project. Built with too much coffee and a genuine curiosity about what *E. coli* would sound like if you listened to it.

