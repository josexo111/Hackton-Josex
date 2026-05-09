import math
import wave
import struct
from typing import Generator

import numpy as np

from chaos_engine import ChaosVector, GlitchState


# ── Frequency Map: DNA Bases → Musical Frequencies ────────────────────────────
#
# Tuning rationale:
#   A = 440.00 Hz  (concert pitch, A4)
#   T = 293.66 Hz  (D4, a minor sixth below A — "cold" interval)
#   C = 523.25 Hz  (C5, just above A — "bright" color)
#   G = 392.00 Hz  (G4, perfect fifth above C — "stable" anchor)
#
# These frequencies form a dm7 chord structure (D-F-A-C) when all four bases
# sound together — chosen because the minor seventh evokes biological tension.
# N (ambiguous) = 261.63 Hz (C4, middle C — a neutral "unknown" signal).

BASE_FREQUENCIES: dict[str, float] = {
    "A": 440.00,   # Adenine  → A4
    "T": 293.66,   # Thymine  → D4
    "C": 523.25,   # Cytosine → C5
    "G": 392.00,   # Guanine  → G4
    "N": 261.63,   # Unknown  → C4 (middle C)
}

# Harmonic series for Additive Synthesis (codon Bio-Chords)
# Each partial is a frequency multiplier + amplitude weight
# Format: [(harmonic_multiplier, relative_amplitude), ...]
SINE_PARTIALS    = [(1.0, 1.0)]                                   # pure fundamental
SQUARE_PARTIALS  = [(1, 1.0), (3, 0.33), (5, 0.20), (7, 0.14)]  # odd harmonics
SAWTOOTH_PARTIALS = [(1, 1.0), (2, 0.50), (3, 0.33), (4, 0.25),  # all harmonics
                     (5, 0.20), (6, 0.17), (7, 0.14), (8, 0.13)]

SAMPLE_RATE = 44100  # Hz — CD quality


# ── Core Oscillator Functions ─────────────────────────────────────────────────

def generate_sine(
    frequency: float,
    duration_s: float,
    amplitude: float = 1.0,
    phase_offset: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a sine wave buffer.

    Args:
        frequency:    Fundamental frequency in Hz.
        duration_s:   Duration in seconds.
        amplitude:    Peak amplitude [0.0, 1.0].
        phase_offset: Starting phase in radians (enables continuous phase stitching).
        sample_rate:  Samples per second.

    Returns:
        np.ndarray: float64 buffer, range [-amplitude, +amplitude].
    """
    t = np.linspace(
        phase_offset / (2 * np.pi * frequency),
        phase_offset / (2 * np.pi * frequency) + duration_s,
        int(sample_rate * duration_s),
        endpoint=False,
        dtype=np.float64,
    )
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def generate_square(
    frequency: float,
    duration_s: float,
    amplitude: float = 1.0,
    n_harmonics: int = 15,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a square wave via Fourier synthesis (band-limited, no aliasing).

    A naïve np.sign(sin(t)) square wave has infinite harmonics and causes
    aliasing distortion at the Nyquist boundary. This implementation builds
    the wave by summing the first `n_harmonics` odd partials of the Fourier
    series, which limits the spectral content to below sample_rate/2.

    Formula:
        x(t) = (4/π) * Σ sin((2k-1)*ωt) / (2k-1)   for k = 1..n_harmonics

    Args:
        frequency:   Fundamental frequency in Hz.
        duration_s:  Duration in seconds.
        amplitude:   Peak amplitude [0.0, 1.0].
        n_harmonics: Number of odd partials to sum. More = sharper edges.
        sample_rate: Audio sample rate.

    Returns:
        np.ndarray: Band-limited square wave, float64.
    """
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float64)
    wave_out = np.zeros(n_samples, dtype=np.float64)

    for k in range(1, n_harmonics + 1):
        harmonic = 2 * k - 1  # odd harmonics: 1, 3, 5, 7 ...
        wave_out += np.sin(2.0 * np.pi * harmonic * frequency * t) / harmonic

    return amplitude * (4.0 / np.pi) * wave_out


def generate_sawtooth(
    frequency: float,
    duration_s: float,
    amplitude: float = 1.0,
    n_harmonics: int = 20,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a sawtooth wave via Fourier synthesis (band-limited).

    Sawtooth = all harmonics with descending amplitude (1/n).
    Maximally complex waveform — used for CRITICAL GLITCH state.
    Contains all integer partials, giving it the harshest timbre.

    Formula:
        x(t) = (2/π) * Σ (-1)^(k+1) * sin(k*ωt) / k   for k = 1..n_harmonics

    Args:
        frequency:   Fundamental frequency in Hz.
        duration_s:  Duration in seconds.
        amplitude:   Peak amplitude [0.0, 1.0].
        n_harmonics: Number of all-integer partials to sum.
        sample_rate: Audio sample rate.

    Returns:
        np.ndarray: Band-limited sawtooth wave, float64.
    """
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float64)
    wave_out = np.zeros(n_samples, dtype=np.float64)

    for k in range(1, n_harmonics + 1):
        sign = (-1) ** (k + 1)
        wave_out += sign * np.sin(2.0 * np.pi * k * frequency * t) / k

    return amplitude * (2.0 / np.pi) * wave_out


# ── Hanning Window Smoother ────────────────────────────────────────────────────

def apply_hanning_envelope(
    buffer: np.ndarray,
    fade_samples: int | None = None,
) -> np.ndarray:
    """
    Apply a Hanning (Hann) window envelope to a waveform buffer.

    PURPOSE — Preventing audio clipping and click artifacts:
    ─────────────────────────────────────────────────────────
    When concatenating two audio buffers (e.g., transitioning from an A-note
    to a T-note), a phase discontinuity at the junction creates a high-frequency
    "click" transient. This is the time-domain equivalent of spectral leakage.

    The Hanning window tapers both ends of each buffer to zero, ensuring smooth
    amplitude transitions: the tail of buffer[n] fades to 0 just as the head of
    buffer[n+1] fades in from 0. The two zero-crossings align → no click.

    Window function:
        w(n) = 0.5 * (1 - cos(2π * n / (N-1)))   for n = 0..N-1

    Alternative approach (not used here):
        A crossfade could also be implemented by overlapping consecutive buffers
        and summing in the overlap region (OLA — Overlap-Add method). This is
        more RAM-intensive but provides better phase continuity for pitched signals.
        For GEN-GLITCH, the per-buffer Hanning window is preferred for its
        simplicity and because the note durations are short enough (30–100ms)
        that crossfade overhead is unjustified.

    Args:
        buffer:        Input numpy array (any dtype, will be float64 output).
        fade_samples:  If set, only apply window to the first+last N samples
                       (partial fade). If None, apply full Hanning window.

    Returns:
        np.ndarray: float64 array with Hanning envelope applied.
    """
    result = buffer.astype(np.float64)
    n = len(result)

    if n == 0:
        return result

    if fade_samples is None:
        # Full Hanning window over entire buffer
        window = np.hanning(n)
        return result * window
    else:
        # Partial fade: taper only the first and last `fade_samples` samples
        fade_samples = min(fade_samples, n // 2)
        fade_in = np.hanning(fade_samples * 2)[:fade_samples]   # rising half
        fade_out = np.hanning(fade_samples * 2)[fade_samples:]  # falling half

        result[:fade_samples] *= fade_in
        result[-fade_samples:] *= fade_out
        return result


# ── Frequency Modulation ───────────────────────────────────────────────────────

def apply_fm_modulation(
    carrier: np.ndarray,
    carrier_freq: float,
    fm_intensity: float,
    modulator_freq: float | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Apply Frequency Modulation (FM) synthesis to a carrier buffer.

    FM Synthesis fundamentals:
        carrier(t) = A * sin(2π * fc * t + β * sin(2π * fm * t))
        where:
            fc = carrier frequency (base DNA note frequency)
            fm = modulator frequency (derived from entropy / GC-skew)
            β  = modulation index (= fm_intensity, controls spectral bandwidth)

    At β = 0: no modulation, pure carrier.
    At β = 1: moderate sidebands, warm vibrato effect.
    At β ≥ 3: heavy sidebands, metallic / glitch timbre (CRITICAL state).

    The modulator frequency defaults to carrier_freq * 0.25 (sub-harmonic)
    to produce a slow pitch wobble. In CRITICAL state, higher fm_intensity
    values push into inharmonic territory.

    Args:
        carrier:        Input waveform buffer (float64).
        carrier_freq:   Fundamental frequency of the carrier in Hz.
        fm_intensity:   Modulation index β [0.0, 1.0 → 0.0, 5.0 internally].
        modulator_freq: FM modulator frequency in Hz. Defaults to carrier/4.
        sample_rate:    Audio sample rate.

    Returns:
        np.ndarray: FM-modulated waveform, float64.
    """
    n_samples = len(carrier)
    if n_samples == 0 or fm_intensity == 0.0:
        return carrier.copy()

    if modulator_freq is None:
        modulator_freq = carrier_freq * 0.25  # sub-harmonic modulator

    t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
    beta = fm_intensity * 5.0  # scale [0,1] → [0,5] modulation index
    modulator = np.sin(2.0 * np.pi * modulator_freq * t)
    phase_mod = beta * modulator

    # Re-synthesize carrier with modulated phase
    modulated = np.sin(2.0 * np.pi * carrier_freq * t + phase_mod)

    # Preserve original amplitude envelope
    rms_original = np.sqrt(np.mean(carrier ** 2)) + 1e-9
    rms_modulated = np.sqrt(np.mean(modulated ** 2)) + 1e-9
    modulated *= (rms_original / rms_modulated)

    return modulated.astype(np.float64)


# ── Additive Synthesis: Bio-Chords ────────────────────────────────────────────

def generate_bio_chord(
    codon: str,
    duration_s: float,
    waveform_type: str = "sine",
    chord_weight: float = 0.5,
    amplitude: float = 0.7,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a Bio-Chord by additive synthesis of a three-nucleotide codon.

    When a start codon (ATG) or any three-base codon is detected in the
    sequence, we don't play three separate notes — we synthesize them
    simultaneously as an additive chord. Each base contributes its fundamental
    frequency, harmonically weighted by the `chord_weight` parameter.

    Additive Synthesis Process:
        1. Look up frequency for each of the 3 codon bases (e.g. ATG → A+T+G)
        2. Generate a waveform for each at its frequency
        3. Mix them together weighted by position in codon:
           - Position 0 (first base):  weight = 1.0 (root of chord)
           - Position 1 (second base): weight = 0.7 (third / fifth)
           - Position 2 (third base):  weight = chord_weight (wobble depth)
        4. Apply Hanning window to remove click artifacts

    Args:
        codon:        3-character nucleotide string (e.g. "ATG", "GCT").
        duration_s:   Duration of the chord event in seconds.
        waveform_type: "sine" | "square" | "sawtooth" — set by Chaos Engine.
        chord_weight:  Amplitude weight for the third partial [0.0, 1.0].
        amplitude:     Master output gain.
        sample_rate:   Audio sample rate.

    Returns:
        np.ndarray: Mixed Bio-Chord waveform, float64.
    """
    if len(codon) != 3:
        codon = (codon + "NNN")[:3]  # pad or truncate to 3

    codon_upper = codon.upper()
    generator = {
        "sine":     generate_sine,
        "square":   generate_square,
        "sawtooth": generate_sawtooth,
    }.get(waveform_type, generate_sine)

    n_samples = int(sample_rate * duration_s)
    chord_buffer = np.zeros(n_samples, dtype=np.float64)

    weights = [1.0, 0.7, chord_weight]

    for i, base in enumerate(codon_upper):
        freq = BASE_FREQUENCIES.get(base, BASE_FREQUENCIES["N"])
        partial_amp = amplitude * weights[i]

        if waveform_type == "sine":
            wave_buf = generate_sine(freq, duration_s, partial_amp, sample_rate=sample_rate)
        elif waveform_type == "square":
            wave_buf = generate_square(freq, duration_s, partial_amp, sample_rate=sample_rate)
        else:
            wave_buf = generate_sawtooth(freq, duration_s, partial_amp, sample_rate=sample_rate)

        chord_buffer += wave_buf

    # Normalize chord to prevent inter-partial clipping, then apply Hanning
    peak = np.max(np.abs(chord_buffer))
    if peak > 0:
        chord_buffer /= peak
        chord_buffer *= amplitude

    return apply_hanning_envelope(chord_buffer, fade_samples=int(sample_rate * 0.005))


# ── PCM Normalizer & WAV Export ───────────────────────────────────────────────

def normalize_to_16bit_pcm(
    audio_float: np.ndarray,
    headroom_db: float = -1.0,
) -> np.ndarray:
    """
    Normalize a float64 audio buffer to 16-bit PCM int16.

    BIT-DEPTH CALIBRATION:
    ──────────────────────
    The numpy synthesis pipeline operates in float64 for maximum numeric
    precision during the computation chain (FM modulation, additive mixing,
    windowing). However, .wav files require integer PCM samples.

    16-bit PCM range: -32768 to +32767 (signed int16, 65536 quantization levels)
    float64 range:    typically -1.0 to +1.0 after normalization

    Conversion steps:
        1. Peak normalization: find max(|x|), scale so that peak = 0 dBFS - headroom
           headroom_db = -1.0 means peak = 10^(-1/20) ≈ 0.891 of full scale
           This prevents inter-sample peaks (true-peak clipping) post-conversion.
        2. Scale to 16-bit range: multiply by 32767.0 (not 32768 — avoids +1 overflow)
        3. Cast to int16: np.int16 clips values outside [-32768, 32767]

    Why headroom?
        After the float→int cast, the sample values are discrete integers.
        A value of exactly 32767.0 rounds fine, but 32767.6 would need to clip
        to 32767. The -1dB headroom gives a 0.109× safety margin against this.
        Also, some DACs and codec pipelines add a small offset during playback;
        headroom absorbs that without audible distortion.

    Args:
        audio_float:  float64 numpy array, any amplitude range.
        headroom_db:  Peak headroom in dBFS (negative = headroom below 0dBFS).

    Returns:
        np.ndarray: int16 numpy array, range [-32768, 32767].
    """
    if len(audio_float) == 0:
        return np.zeros(0, dtype=np.int16)

    # Compute target peak level
    headroom_linear = 10.0 ** (headroom_db / 20.0)  # e.g. -1dB → 0.8913

    # Find current peak
    peak = np.max(np.abs(audio_float))
    if peak == 0.0:
        return np.zeros(len(audio_float), dtype=np.int16)

    # Scale to target peak level
    normalized = audio_float * (headroom_linear / peak)

    # Scale to 16-bit integer range and cast
    pcm_float = normalized * 32767.0
    pcm_int16 = np.clip(pcm_float, -32768, 32767).astype(np.int16)

    return pcm_int16


def export_wav(
    pcm_data: np.ndarray,
    filepath: str,
    sample_rate: int = SAMPLE_RATE,
    n_channels: int = 1,
) -> None:
    """
    Write a 16-bit PCM .wav file from a numpy int16 buffer.

    Uses Python's stdlib `wave` module — no external audio library required.
    The .wav format is RIFF-WAVE with little-endian byte order.

    WAV Header parameters:
        nchannels:    1 (mono) or 2 (stereo)
        sampwidth:    2 bytes = 16-bit PCM
        framerate:    sample_rate (default 44100 Hz)
        nframes:      len(pcm_data) (for mono), len(pcm_data)//2 (for stereo)

    Args:
        pcm_data:    int16 numpy array (mono: shape (N,), stereo: shape (N,2))
        filepath:    Output .wav file path.
        sample_rate: Audio sample rate in Hz.
        n_channels:  1 for mono, 2 for stereo.
    """
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)          # 2 bytes = 16-bit
        wf.setframerate(sample_rate)
        # Convert numpy int16 array to raw bytes (little-endian)
        wf.writeframes(pcm_data.tobytes())


# ── Main Audio Synthesis Pipeline ─────────────────────────────────────────────

class SonicSynthesizer:
    """
    Stateful DSP pipeline that consumes ChaosVectors and produces a
    unified audio buffer, which is then exported as a 16-bit WAV file.

    The synthesizer maintains a phase accumulator across consecutive
    nucleotide notes to ensure smooth pitch transitions (phase continuity).

    Synchronization contract with Module D:
    ───────────────────────────────────────
    Each ChaosVector is associated with a specific nucleotide position.
    The synthesizer assigns each vector a fixed `NOTE_DURATION_MS` worth of
    audio samples. Module D uses the same position index to synchronize
    which terminal row is "active" during playback.

    To achieve real-time sync:
        audio_position_s = position_index * NOTE_DURATION_MS / 1000.0
        terminal_row     = position_index  (direct 1:1 mapping)
    Both are driven by the same position clock — derived from the k-mer index.
    """

    NOTE_DURATION_MS = 80    # Duration per nucleotide event in milliseconds
    CODON_TRIGGER_FREQ = 3   # Trigger Bio-Chord every N nucleotides (=1 codon)

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._buffers: list[np.ndarray] = []
        self._phase_accumulator: float = 0.0
        self._note_count = 0

    def process_vector(
        self,
        vector: ChaosVector,
        base: str = "A",
        codon: str | None = None,
    ) -> np.ndarray:
        """
        Process one ChaosVector and return the corresponding audio buffer.

        If a codon string is provided and chord_weight > 0.3, a Bio-Chord
        is generated instead of a monophonic note.

        Args:
            vector: ChaosVector from Chaos Engine.
            base:   Current nucleotide base (A/T/C/G/N).
            codon:  Three-base codon string, if currently in a codon.

        Returns:
            np.ndarray: float64 audio buffer for this event.
        """
        duration_s = self.NOTE_DURATION_MS / 1000.0
        freq = BASE_FREQUENCIES.get(base.upper(), BASE_FREQUENCIES["N"])

        # Apply detune from GC-skew (cents → frequency ratio)
        if vector.detune_cents != 0.0:
            freq *= 2.0 ** (vector.detune_cents / 1200.0)

        # ── Generate base waveform ─────────────────────────────────────────────
        if codon and len(codon) == 3 and vector.chord_weight > 0.3:
            # Bio-Chord synthesis (codon detected)
            buffer = generate_bio_chord(
                codon=codon,
                duration_s=duration_s,
                waveform_type=vector.waveform_type,
                chord_weight=vector.chord_weight,
                amplitude=vector.amplitude_envelope,
                sample_rate=self.sample_rate,
            )
        else:
            # Monophonic synthesis
            if vector.waveform_type == "sine":
                buffer = generate_sine(freq, duration_s, vector.amplitude_envelope,
                                       self._phase_accumulator, self.sample_rate)
            elif vector.waveform_type == "square":
                buffer = generate_square(freq, duration_s, vector.amplitude_envelope,
                                         sample_rate=self.sample_rate)
            else:  # sawtooth — CRITICAL GLITCH
                buffer = generate_sawtooth(freq, duration_s, vector.amplitude_envelope,
                                           sample_rate=self.sample_rate)

            # Apply FM modulation from entropy LFO
            if vector.fm_intensity > 0.05:
                buffer = apply_fm_modulation(buffer, freq, vector.fm_intensity,
                                             sample_rate=self.sample_rate)

            # Apply Hanning window for click-free transitions
            buffer = apply_hanning_envelope(
                buffer,
                fade_samples=int(self.sample_rate * 0.003),  # 3ms fade
            )

        # Update phase accumulator for continuity
        n_samples = len(buffer)
        self._phase_accumulator = (
            2.0 * np.pi * freq * n_samples / self.sample_rate
        ) % (2.0 * np.pi)

        self._buffers.append(buffer)
        self._note_count += 1
        return buffer

    def render(self) -> np.ndarray:
        """
        Concatenate all processed buffers into a single float64 audio array.

        Returns:
            np.ndarray: Complete float64 audio signal.
        """
        if not self._buffers:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(self._buffers)

    def export(self, filepath: str, headroom_db: float = -1.0) -> str:
        """
        Render, normalize to 16-bit PCM, and export as .wav file.

        Args:
            filepath:     Output .wav file path.
            headroom_db:  Peak headroom for PCM normalization.

        Returns:
            str: Filepath of exported file.
        """
        raw_audio = self.render()
        pcm = normalize_to_16bit_pcm(raw_audio, headroom_db)
        export_wav(pcm, filepath, self.sample_rate)
        duration_s = len(raw_audio) / self.sample_rate
        print(
            f"[SONIC] Exported: {filepath} | "
            f"{self._note_count} events | "
            f"{duration_s:.2f}s | "
            f"{len(pcm):,} samples | 16-bit PCM @ {self.sample_rate}Hz"
        )
        return filepath