import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Generator

from bio_kernel import BioKernelReport, KmerWindow, OpenReadingFrame


# ── Glitch State Machine ───────────────────────────────────────────────────────

class GlitchState(Enum):
    """
    The four states of the Chaos Engine state machine.
    Transitions are entropy-driven, not time-driven.
    """
    SILENT   = auto()   # H(s) < 0.5  — homopolymer run, dead signal
    STABLE   = auto()   # H(s) 0.5–1.4 — sine-wave territory
    TURBULENT = auto()  # H(s) 1.4–1.9 — square-wave / harmonic distortion
    CRITICAL  = auto()  # H(s) > 1.9  — FULL GLITCH. Sawtooth + visual inversion


ENTROPY_THRESHOLDS = {
    GlitchState.SILENT:    (0.0, 0.5),
    GlitchState.STABLE:    (0.5, 1.4),
    GlitchState.TURBULENT: (1.4, 1.9),
    GlitchState.CRITICAL:  (1.9, 2.01),  # theoretical max for 4-letter alphabet = 2.0
}

CRITICAL_GLITCH_THRESHOLD = 1.9


# ── Chaos Vector ───────────────────────────────────────────────────────────────

@dataclass
class ChaosVector:
    """
    The fundamental output unit of the Chaos Engine.
    One ChaosVector is generated per k-mer window and consumed by:
      - Module C (Sonic Synthesis): fm_intensity, waveform_type, chord_weight
      - Module D (Glitch Visualizer): visual_density, color_shift, invert_flag

    All parameters are normalized to [0.0, 1.0] unless otherwise noted.

    Coordinate system analogy (DSP):
        Think of this as a point in a 6D parameter space where each axis
        controls one dimension of the audiovisual output. The Chaos Engine
        computes these coordinates from genomic signal; the output modules
        render them. Decoupled by design.
    """
    # ── Identity ───────────────────────────────────────────────────────────────
    position: int           # absolute nucleotide position in sequence
    sequence_fragment: str  # the raw k-mer for reference

    # ── Entropy State ──────────────────────────────────────────────────────────
    entropy: float          # raw H(s) value [0.0, 2.0]
    glitch_state: GlitchState
    is_critical: bool = False

    # ── Audio Parameters (consumed by Module C) ────────────────────────────────
    fm_intensity: float = 0.0       # frequency modulation depth [0.0, 1.0]
    waveform_type: str = "sine"     # "sine" | "square" | "sawtooth"
    chord_weight: float = 0.0       # additive synthesis blend [0.0, 1.0]
    amplitude_envelope: float = 1.0 # output gain [0.0, 1.0]
    detune_cents: float = 0.0       # pitch microtonal offset in cents

    # ── Visual Parameters (consumed by Module D) ───────────────────────────────
    visual_density: float = 0.5     # ASCII glyph density [0.0, 1.0]
    color_shift: int = 0            # ANSI color code offset [0, 255]
    horizontal_jitter: int = 0      # terminal column displacement in chars
    invert_colors: bool = False      # full ANSI color inversion
    gc_glow: bool = False            # electric cyan highlight (high GC)

    # ── ORF Annotation ─────────────────────────────────────────────────────────
    in_orf: bool = False
    orf_frame: int = -1


@dataclass
class GlitchEvent:
    """
    Broadcast when GlitchState.CRITICAL is entered.
    Consumed by Module D for terminal visual effects and by Module C
    for waveform switching without clicks (requires crossfade buffer).
    """
    position: int
    entropy: float
    vector: ChaosVector
    duration_samples: int = 0  # set by Module C based on sample rate


# ── Chaos Engine Core ──────────────────────────────────────────────────────────

class ChaosEngine:
    """
    Stateful transduction engine. Maintains a state machine and outputs
    a stream of ChaosVectors for the downstream modules to consume.

    Usage:
        engine = ChaosEngine(sample_rate=44100)
        for vector in engine.transduce(bio_report):
            audio_module.process(vector)
            visual_module.render(vector)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        critical_threshold: float = CRITICAL_GLITCH_THRESHOLD,
        hotspot_fm_boost: float = 0.3,
    ):
        """
        Args:
            sample_rate:        Audio sample rate in Hz. Affects duration_samples.
            critical_threshold: H(s) value that triggers CRITICAL state.
            hotspot_fm_boost:   Additional FM intensity added to hotspot regions.
        """
        self.sample_rate = sample_rate
        self.critical_threshold = critical_threshold
        self.hotspot_fm_boost = hotspot_fm_boost

        self._current_state = GlitchState.SILENT
        self._glitch_events: list[GlitchEvent] = []
        self._critical_count = 0

    @property
    def glitch_events(self) -> list[GlitchEvent]:
        """All CRITICAL GLITCH events detected during the current transduction."""
        return self._glitch_events

    @property
    def critical_count(self) -> int:
        return self._critical_count

    def _classify_entropy(self, entropy: float) -> GlitchState:
        """Classify entropy value into a GlitchState."""
        for state, (lo, hi) in ENTROPY_THRESHOLDS.items():
            if lo <= entropy < hi:
                return state
        return GlitchState.CRITICAL  # above 2.0 is theoretically impossible but safe

    def _compute_chaos_vector(
        self,
        window: KmerWindow,
        orf_map: dict[int, OpenReadingFrame],
    ) -> ChaosVector:
        """
        Core transduction function: maps one k-mer window to one ChaosVector.

        Parameter Mapping Logic:
        ─────────────────────────
        fm_intensity    = entropy / 2.0  (linear scale, 0.0–1.0)
        chord_weight    = orf presence bonus + entropy contribution
        visual_density  = mapped nonlinearly (low entropy → sparse glyphs)
        color_shift     = GC-skew mapped to ANSI 256-color palette offset
        horizontal_jitter = 0 in STABLE, random in TURBULENT, max in CRITICAL

        Waveform Selection:
          SILENT    → sine (gentle carrier, near-zero amplitude)
          STABLE    → sine
          TURBULENT → square (odd harmonics, buzzy texture)
          CRITICAL  → sawtooth (all harmonics, full glitch timbre)

        Args:
            window:   KmerWindow from Bio-Kernel.
            orf_map:  Dict mapping position ranges to ORF objects.

        Returns:
            ChaosVector: Complete parameter vector.
        """
        entropy = window.local_entropy
        state = self._classify_entropy(entropy)
        is_critical = entropy >= self.critical_threshold

        # ── FM Intensity: linear entropy-to-modulation mapping ────────────────
        fm_intensity = entropy / 2.0  # normalized [0.0, 1.0]
        if window.is_hotspot:
            fm_intensity = min(1.0, fm_intensity + self.hotspot_fm_boost)

        # ── Waveform selection ────────────────────────────────────────────────
        waveform_map = {
            GlitchState.SILENT:    "sine",
            GlitchState.STABLE:    "sine",
            GlitchState.TURBULENT: "square",
            GlitchState.CRITICAL:  "sawtooth",
        }
        waveform_type = waveform_map[state]

        # ── Chord weight: ORF presence amplifies harmonic richness ────────────
        in_orf = window.start in orf_map
        orf_frame = orf_map[window.start].frame if in_orf else -1
        chord_weight = (entropy / 2.0) * 0.6 + (0.4 if in_orf else 0.0)

        # ── Amplitude: silence low-entropy, boost critical ────────────────────
        amplitude = max(0.1, entropy / 2.0) if state != GlitchState.SILENT else 0.05

        # ── Detune: GC-Skew introduces microtonal pitch variation ─────────────
        # Skew of ±1.0 maps to ±50 cents of detune
        detune_cents = window.gc_skew * 50.0

        # ── Visual density: nonlinear mapping (perceptual) ────────────────────
        # Low entropy = sparse dots; critical entropy = dense block chars
        visual_density = (entropy / 2.0) ** 0.7  # gamma 0.7 → perceptual curve

        # ── Color shift: GC-skew → ANSI 256-color palette index ──────────────
        # GC-skew [-1, 1] → palette offset [0, 50]
        color_shift = int((window.gc_skew + 1.0) / 2.0 * 50)

        # ── Horizontal jitter: escalates with glitch state ────────────────────
        jitter_map = {
            GlitchState.SILENT: 0,
            GlitchState.STABLE: 0,
            GlitchState.TURBULENT: _deterministic_jitter(window.start, max_val=3),
            GlitchState.CRITICAL: _deterministic_jitter(window.start, max_val=8),
        }
        jitter = jitter_map[state]

        # ── GC glow: electric cyan for high-GC regions ────────────────────────
        gc_content = (window.sequence.count("G") + window.sequence.count("C"))
        gc_content /= max(len(window.sequence), 1)
        gc_glow = gc_content > 0.65  # top ~35% of GC distribution

        vector = ChaosVector(
            position=window.start,
            sequence_fragment=window.sequence[:8] + "...",
            entropy=entropy,
            glitch_state=state,
            is_critical=is_critical,
            fm_intensity=round(fm_intensity, 4),
            waveform_type=waveform_type,
            chord_weight=round(chord_weight, 4),
            amplitude_envelope=round(amplitude, 4),
            detune_cents=round(detune_cents, 2),
            visual_density=round(visual_density, 4),
            color_shift=color_shift,
            horizontal_jitter=jitter,
            invert_colors=is_critical,
            gc_glow=gc_glow,
            in_orf=in_orf,
            orf_frame=orf_frame,
        )

        # ── CRITICAL GLITCH event broadcast ───────────────────────────────────
        if is_critical and self._current_state != GlitchState.CRITICAL:
            # Compute duration: number of audio samples for this k-mer window
            # Assumes each nucleotide = 1 sample at current note duration
            event = GlitchEvent(
                position=window.start,
                entropy=entropy,
                vector=vector,
                duration_samples=len(window.sequence),
            )
            self._glitch_events.append(event)
            self._critical_count += 1

        self._current_state = state
        return vector

    def transduce(
        self,
        report: BioKernelReport,
    ) -> Generator[ChaosVector, None, None]:
        """
        Primary transduction pipeline.
        Converts a BioKernelReport into a stream of ChaosVectors.

        This generator is the main interface between Bio-Kernel and the
        audio/visual output modules. Callers iterate over it and route
        each vector to Module C and Module D simultaneously.

        Args:
            report: BioKernelReport from Module A (Bio-Kernel).

        Yields:
            ChaosVector: One per k-mer window in the report.
        """
        # Build ORF position lookup: position → ORF
        orf_map: dict[int, OpenReadingFrame] = {}
        for orf in report.orfs:
            for pos in range(orf.start, orf.stop, 3):
                orf_map[pos] = orf

        for window in report.kmer_windows:
            vector = self._compute_chaos_vector(window, orf_map)
            yield vector

    def get_summary(self) -> dict:
        """
        Returns a human-readable summary of the transduction session.
        Use for logging, UI headers, and debug output.
        """
        return {
            "critical_glitch_events": self._critical_count,
            "glitch_event_positions": [e.position for e in self._glitch_events[:5]],
            "final_state": self._current_state.name,
        }


# ── Utility Functions ──────────────────────────────────────────────────────────

def _deterministic_jitter(seed: int, max_val: int) -> int:
    """
    Deterministic pseudo-random jitter based on sequence position.
    NOT random.randint — deterministic ensures reproducible output
    and avoids race conditions in the real-time rendering loop.

    Uses a simple hash: LCG-derived scramble of the position seed.
    """
    scrambled = (seed * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative hash
    return (scrambled % (max_val * 2 + 1)) - max_val  # range: [-max_val, max_val]


def normalize_entropy_stream(
    entropies: list[float],
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> list[float]:
    """
    Min-max normalize a stream of entropy values.
    Used when exporting entropy curves as audio control signals (LFO).

    Args:
        entropies:   Raw H(s) values from Bio-Kernel.
        target_min:  Output range minimum.
        target_max:  Output range maximum.

    Returns:
        Normalized float list in [target_min, target_max].
    """
    if not entropies:
        return []
    e_min = min(entropies)
    e_max = max(entropies)
    if e_max == e_min:
        return [target_min] * len(entropies)
    span = e_max - e_min
    return [
        target_min + (e - e_min) / span * (target_max - target_min)
        for e in entropies
    ]

