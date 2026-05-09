 Synchronization with Audio (Module C):
───────────────────────────────────────
Terminal rendering rate matches audio sample consumption rate.
  audio_note_duration = 80ms (NOTE_DURATION_MS in sonic_synthesis.py)
  terminal_frame_rate = 1000 / 80 = 12.5 Hz (new row every 80ms)

For real-time sync:
  time.sleep(NOTE_DURATION_MS / 1000.0)  between row renders
  This aligns visual rows with audio event timing via the same position clock.
  A production implementation would use a shared threading.Event or
  a circular buffer with a producer (audio) / consumer (visual) pair
  synchronized through a read pointer updated every audio callback.

Glyph Density Mapping (visual_density → ASCII character set):
  0.0 – 0.2 → " "  (space, emptiness)
  0.2 – 0.4 → "."  (near-nothing)
  0.4 – 0.6 → ":"  (low texture)
  0.6 – 0.7 → "~"  (mild complexity)
  0.7 – 0.8 → "*"  (moderate signal)
  0.8 – 0.9 → "#"  (high complexity)
  0.9 – 1.0 → "@"  (critical density, maximum glitch)
"""

import os
import sys
import time
import shutil
from dataclasses import dataclass
from typing import Callable

from chaos_engine import ChaosVector, GlitchState


# ── ANSI Escape Codes ──────────────────────────────────────────────────────────

class ANSI:
    """
    ANSI escape code constants.
    All codes use the CSI (Control Sequence Introducer) prefix: \033[
    """
    RESET        = "\033[0m"
    BOLD         = "\033[1m"
    DIM          = "\033[2m"
    INVERT       = "\033[7m"
    BLINK        = "\033[5m"

    # Standard colors
    BLACK        = "\033[30m"
    RED          = "\033[31m"
    GREEN        = "\033[32m"
    YELLOW       = "\033[33m"
    BLUE         = "\033[34m"
    MAGENTA      = "\033[35m"
    CYAN         = "\033[36m"
    WHITE        = "\033[37m"

    # Bright variants
    BRIGHT_GREEN  = "\033[92m"
    BRIGHT_CYAN   = "\033[96m"  # Electric Cyan — high-GC glow
    BRIGHT_WHITE  = "\033[97m"
    BRIGHT_RED    = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_MAGENTA = "\033[95m"

    # Background
    BG_BLACK     = "\033[40m"
    BG_RED       = "\033[41m"
    BG_GREEN     = "\033[42m"
    BG_CYAN      = "\033[46m"
    BG_WHITE     = "\033[47m"
    BG_BRIGHT_RED = "\033[101m"

    # 256-color support: \033[38;5;{n}m for foreground
    @staticmethod
    def color256(n: int) -> str:
        """256-color foreground. n in range [0, 255]."""
        return f"\033[38;5;{n % 256}m"

    @staticmethod
    def bg_color256(n: int) -> str:
        """256-color background."""
        return f"\033[48;5;{n % 256}m"

    @staticmethod
    def cursor_move(row: int, col: int) -> str:
        """Move cursor to absolute position (1-indexed)."""
        return f"\033[{row};{col}H"

    @staticmethod
    def clear_line() -> str:
        return "\033[2K"

    @staticmethod
    def hide_cursor() -> str:
        return "\033[?25l"

    @staticmethod
    def show_cursor() -> str:
        return "\033[?25h"

    @staticmethod
    def clear_screen() -> str:
        return "\033[2J\033[H"


# ── Nucleotide Color Profiles ─────────────────────────────────────────────────

BASE_COLORS: dict[str, str] = {
    "A": ANSI.BRIGHT_GREEN,    # Adenine   → Matrix green
    "T": ANSI.YELLOW,          # Thymine   → warm amber/yellow
    "C": ANSI.BRIGHT_CYAN,     # Cytosine  → electric cyan (purine partner)
    "G": ANSI.BRIGHT_MAGENTA,  # Guanine   → magenta (complement of cyan)
    "N": ANSI.DIM + ANSI.WHITE, # Unknown  → dim white (noise floor)
}

# Glyph density map: visual_density → ASCII character
DENSITY_GLYPHS = [
    (0.0,  " "),
    (0.2,  "·"),
    (0.35, ":"),
    (0.5,  "~"),
    (0.62, "*"),
    (0.75, "+"),
    (0.85, "#"),
    (0.92, "$"),
    (0.97, "@"),
]

def density_to_glyph(density: float) -> str:
    """Map visual_density [0.0, 1.0] to ASCII glyph."""
    glyph = " "
    for threshold, char in DENSITY_GLYPHS:
        if density >= threshold:
            glyph = char
    return glyph


# ── Waterfall Row ─────────────────────────────────────────────────────────────

@dataclass
class WaterfallRow:
    """
    One rendered line in the terminal waterfall.
    Stored in a circular buffer for scrolling effect.
    """
    position: int
    content: str     # rendered ANSI string (pre-composed)
    raw_sequence: str
    glitch_state: GlitchState
    is_critical: bool
    timestamp: float


# ── Glitch Visualizer ─────────────────────────────────────────────────────────

class GlitchVisualizer:
    """
    Terminal-based DNA waterfall visualizer.

    Architecture:
        Maintains a circular row buffer of rendered ANSI lines.
        Each call to render_vector() adds a new row to the buffer.
        The display() method outputs the buffer to stdout as a scrolling feed.

    Synchronization with Audio:
        The note_duration_s parameter must match SonicSynthesizer.NOTE_DURATION_MS.
        Each render_vector() call sleeps for note_duration_s before returning,
        creating a frame-locked timing that keeps terminal rows and audio events
        in sync. For production, replace time.sleep() with a condition variable
        triggered by the audio callback thread.

    Usage:
        viz = GlitchVisualizer(terminal_width=80, note_duration_ms=80)
        viz.start()
        for vector, base in zip(chaos_vectors, bases):
            viz.render_vector(vector, base)
        viz.stop()
    """

    def __init__(
        self,
        terminal_width: int | None = None,
        note_duration_ms: int = 80,
        show_stats: bool = True,
        use_rich: bool = False,
    ):
        """
        Args:
            terminal_width:   Override terminal width. None = auto-detect.
            note_duration_ms: Frame duration in ms (must match audio note length).
            show_stats:       Show entropy/state HUD in the header.
            use_rich:         Use Rich library if available (fallback to ANSI).
        """
        term_size = shutil.get_terminal_size(fallback=(80, 24))
        self.width = terminal_width or term_size.columns
        self.height = term_size.lines
        self.note_duration_s = note_duration_ms / 1000.0
        self.show_stats = show_stats

        self._row_buffer: list[WaterfallRow] = []
        self._critical_count = 0
        self._total_rows = 0
        self._running = False

        # Rich integration (optional)
        self._rich_console = None
        if use_rich:
            try:
                from rich.console import Console
                self._rich_console = Console()
            except ImportError:
                pass

    def start(self) -> None:
        """Initialize terminal display."""
        self._running = True
        sys.stdout.write(ANSI.hide_cursor())
        sys.stdout.write(ANSI.clear_screen())
        self._render_header()
        sys.stdout.flush()

    def stop(self) -> None:
        """Restore terminal to normal state."""
        self._running = False
        sys.stdout.write(ANSI.show_cursor())
        sys.stdout.write(ANSI.RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render_header(self) -> None:
        """Render the static header bar at the top of the terminal."""
        title = " GEN-GLITCH :: BIO-HACKER DNA TRANSDUCTION ENGINE "
        padding = "═" * ((self.width - len(title)) // 2)
        header_line = f"{padding}{title}{padding}"[:self.width]

        sys.stdout.write(ANSI.cursor_move(1, 1))
        sys.stdout.write(ANSI.color256(51))  # bright cyan
        sys.stdout.write(ANSI.BOLD)
        sys.stdout.write(header_line)
        sys.stdout.write(ANSI.RESET)
        sys.stdout.write("\n")

        col_headers = (
            f"{'POS':>8}  "
            f"{'BASE':^6}  "
            f"{'STATE':^10}  "
            f"{'H(s)':^6}  "
            f"{'GC-SKEW':^8}  "
            f"{'WAVEFORM':^10}  "
            f"{'WATERFALL SIGNAL'}"
        )
        sys.stdout.write(ANSI.color256(242))  # dim gray
        sys.stdout.write(col_headers[:self.width])
        sys.stdout.write(ANSI.RESET)
        sys.stdout.write("\n")
        sys.stdout.write(ANSI.color256(238))
        sys.stdout.write("─" * self.width)
        sys.stdout.write(ANSI.RESET)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _build_waterfall_signal(
        self,
        vector: ChaosVector,
        base: str,
        sequence_fragment: str,
    ) -> str:
        """
        Build the visual "waterfall" signal: a horizontal row of characters
        whose density, color, and jitter reflect the Chaos Vector.

        Layout:
            [jitter spaces] [base char] [density glyph strip] [ORF marker]

        CRITICAL GLITCH effects (applied when vector.is_critical):
          - invert_colors: wraps entire row in ANSI 7m (reverse video)
          - horizontal_jitter: shifts the row by N columns
          - Random 256-color assignments to glitch characters
        """
        available_width = self.width - 42  # reserved for stat columns
        glyph = density_to_glyph(vector.visual_density)

        # Build base nucleotide display
        base_color = BASE_COLORS.get(base.upper(), ANSI.WHITE)

        # Electric Cyan override for high-GC regions
        if vector.gc_glow:
            base_color = ANSI.BRIGHT_CYAN + ANSI.BOLD

        # Build the signal strip
        strip_chars = []

        # Apply horizontal jitter offset
        jitter = abs(vector.horizontal_jitter)
        if jitter > 0:
            strip_chars.append(" " * jitter)

        # Nucleotide character with color
        strip_chars.append(f"{base_color}{base.upper()}{ANSI.RESET}")

        # Density glyph fill
        fill_width = max(0, available_width - jitter - 1 - (3 if vector.in_orf else 0))
        density_fill = _build_density_strip(
            glyph=glyph,
            width=fill_width,
            vector=vector,
            base_color=base_color,
        )
        strip_chars.append(density_fill)

        # ORF frame indicator
        if vector.in_orf:
            frame_colors = [ANSI.YELLOW, ANSI.BRIGHT_CYAN, ANSI.BRIGHT_MAGENTA]
            fc = frame_colors[vector.orf_frame % 3]
            strip_chars.append(f"{fc}[{vector.orf_frame}]{ANSI.RESET}")

        signal = "".join(strip_chars)

        # CRITICAL GLITCH: invert entire row
        if vector.invert_colors:
            signal = f"{ANSI.INVERT}{ANSI.BG_BRIGHT_RED}{signal}{ANSI.RESET}"

        return signal

    def render_vector(
        self,
        vector: ChaosVector,
        base: str = "A",
        sequence_fragment: str = "",
    ) -> None:
        """
        Render one ChaosVector as a terminal row and append to the waterfall.

        This is the primary public method called by the main GEN-GLITCH loop.
        It blocks for note_duration_s to maintain sync with audio playback.

        Args:
            vector:            ChaosVector from Chaos Engine.
            base:              Current nucleotide character.
            sequence_fragment: Short sequence context for display.
        """
        if not self._running:
            return

        # State display string
        state_display = {
            GlitchState.SILENT:    f"{ANSI.DIM}{ANSI.WHITE}SILENT   {ANSI.RESET}",
            GlitchState.STABLE:    f"{ANSI.BRIGHT_GREEN}STABLE   {ANSI.RESET}",
            GlitchState.TURBULENT: f"{ANSI.YELLOW}TURBULENT{ANSI.RESET}",
            GlitchState.CRITICAL:  f"{ANSI.INVERT}{ANSI.BRIGHT_RED}CRITICAL!{ANSI.RESET}",
        }[vector.glitch_state]

        # Entropy display
        entropy_color = ANSI.BRIGHT_GREEN
        if vector.entropy > 1.9:
            entropy_color = ANSI.BRIGHT_RED + ANSI.BOLD
        elif vector.entropy > 1.4:
            entropy_color = ANSI.YELLOW
        entropy_str = f"{entropy_color}{vector.entropy:.4f}{ANSI.RESET}"

        # GC-skew display
        skew_color = ANSI.BRIGHT_CYAN if vector.gc_glow else ANSI.color256(245)
        gc_skew_val = (sequence_fragment.count("G") - sequence_fragment.count("C"))
        gc_skew_denom = sequence_fragment.count("G") + sequence_fragment.count("C")
        gc_skew = gc_skew_val / max(gc_skew_denom, 1)
        skew_str = f"{skew_color}{gc_skew:+.3f}{ANSI.RESET}"

        # Waveform display
        wave_colors = {
            "sine":     ANSI.color256(82),   # green
            "square":   ANSI.color256(226),  # yellow
            "sawtooth": ANSI.BRIGHT_RED,
        }
        wc = wave_colors.get(vector.waveform_type, ANSI.WHITE)
        wave_str = f"{wc}{vector.waveform_type:<10}{ANSI.RESET}"

        # Waterfall signal
        signal_str = self._build_waterfall_signal(vector, base, sequence_fragment)

        # Compose full row
        row_line = (
            f"{ANSI.color256(242)}{vector.position:>8}{ANSI.RESET}  "
            f"{BASE_COLORS.get(base.upper(), ANSI.WHITE)}{base.upper():^6}{ANSI.RESET}  "
            f"{state_display}  "
            f"{entropy_str:^6}  "
            f"{skew_str:^8}  "
            f"{wave_str}  "
            f"{signal_str}"
        )

        # CRITICAL GLITCH: flash header
        if vector.is_critical:
            self._critical_count += 1
            self._flash_critical_event(vector.position, vector.entropy)

        sys.stdout.write(row_line + "\n")
        sys.stdout.flush()

        self._total_rows += 1

        # Frame lock: sleep for note duration to sync with audio
        time.sleep(self.note_duration_s)

    def _flash_critical_event(self, position: int, entropy: float) -> None:
        """
        Display a CRITICAL GLITCH event banner.
        Triggered when H(s) > 1.9 — the most visually disruptive output state.
        """
        banner = (
            f"  ⚡ CRITICAL GLITCH @ pos {position:,} | "
            f"H(s)={entropy:.6f} | FRAMESHIFT DETECTED | "
            f"SAWTOOTH ENGAGED ⚡  "
        )
        banner = banner[:self.width]
        padding = " " * max(0, self.width - len(banner))

        sys.stdout.write(
            f"{ANSI.INVERT}{ANSI.BRIGHT_RED}{ANSI.BLINK}"
            f"{banner}{padding}"
            f"{ANSI.RESET}\n"
        )
        sys.stdout.flush()
        time.sleep(0.15)  # hold the flash briefly


def _build_density_strip(
    glyph: str,
    width: int,
    vector: ChaosVector,
    base_color: str,
) -> str:
    """
    Build a horizontal strip of density glyphs with per-character color variation.

    In CRITICAL state, individual characters get randomized 256-color assignments
    using a deterministic hash of their position (reproducible, not random).
    This creates the "screen corruption" glitch aesthetic without actual randomness.
    """
    if width <= 0:
        return ""

    chars = []
    for i in range(width):
        pos_seed = (vector.position + i) * 31337  # arbitrary prime
        pos_seed = (pos_seed ^ (pos_seed >> 16)) & 0xFFFFFF

        if vector.is_critical:
            # Glitch: deterministic color chaos
            char_color_idx = (pos_seed % 200) + 16  # avoid first 16 system colors
            # Randomize glyphs from the density set
            glyph_chars = "#@$~*:. "
            char_glyph = glyph_chars[pos_seed % len(glyph_chars)]
            chars.append(
                f"{ANSI.color256(char_color_idx)}{char_glyph}{ANSI.RESET}"
            )
        elif vector.glitch_state == GlitchState.TURBULENT:
            # Turbulent: slight color variation
            variation = (pos_seed % 3) - 1
            color_idx = max(0, min(255, vector.color_shift + variation + 82))
            chars.append(f"{ANSI.color256(color_idx)}{glyph}{ANSI.RESET}")
        else:
            # Stable: uniform color
            chars.append(f"{base_color}{glyph}{ANSI.RESET}")

    return "".join(chars)
