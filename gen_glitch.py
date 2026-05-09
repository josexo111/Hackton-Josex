import argparse
import sys
import os
from pathlib import Path

from bio_kernel import run_bio_kernel, BioKernelReport
from chaos_engine import ChaosEngine
from sonic_synthesis import SonicSynthesizer
from glitch_visualizer import GlitchVisualizer


def transduce_sequence(
    fasta_path: str,
    output_wav: str | None = None,
    kmer_size: int = 100,
    kmer_step: int = 25,
    min_orf_bp: int = 150,
    render_audio: bool = True,
    render_visual: bool = True,
    max_sequences: int = 1,
) -> None:
    """
    Main GEN-GLITCH transduction pipeline.

    Processes one or more FASTA records from `fasta_path` and produces:
      - ASCII waterfall terminal display
      - 16-bit PCM .wav audio output

    Args:
        fasta_path:    Input .fasta file path.
        output_wav:    Output .wav file path. Defaults to input basename + .wav.
        kmer_size:     k-mer window size for Bio-Kernel.
        kmer_step:     Stride between k-mer windows.
        min_orf_bp:    Minimum ORF length in base pairs.
        render_audio:  Whether to run Sonic Synthesis.
        render_visual: Whether to run Glitch Visualizer.
        max_sequences: Maximum FASTA records to process.
    """
    if output_wav is None:
        stem = Path(fasta_path).stem
        output_wav = f"{stem}_transduced.wav"

    print(f"\n[GEN-GLITCH] Initiating transduction: {fasta_path}")
    print(f"[GEN-GLITCH] k-mer: {kmer_size}bp | step: {kmer_step}bp | "
          f"min ORF: {min_orf_bp}bp\n")

    # ── Initialize Modules ────────────────────────────────────────────────────
    chaos_engine  = ChaosEngine(sample_rate=44100)
    synthesizer   = SonicSynthesizer(sample_rate=44100) if render_audio else None
    visualizer    = GlitchVisualizer(note_duration_ms=80) if render_visual else None

    if visualizer:
        visualizer.start()

    # ── Process each FASTA record ─────────────────────────────────────────────
    for report in run_bio_kernel(
        filepath=fasta_path,
        kmer_size=kmer_size,
        kmer_step=kmer_step,
        min_orf_bp=min_orf_bp,
        max_sequences=max_sequences,
    ):
        _print_report_header(report)

        # ── Stream ChaosVectors through the pipeline ──────────────────────────
        for idx, vector in enumerate(chaos_engine.transduce(report)):
            # Extract the base at this window's start position
            base = report.kmer_windows[idx].sequence[0] if idx < len(report.kmer_windows) else "N"

            # Extract codon (triplet at this position) if in an ORF
            codon = None
            if vector.in_orf:
                pos = vector.position
                if pos + 3 <= report.total_length:
                    codon = report.kmer_windows[idx].sequence[:3]

            # ── Module C: Audio Synthesis ─────────────────────────────────────
            if synthesizer:
                synthesizer.process_vector(
                    vector=vector,
                    base=base,
                    codon=codon,
                )

            # ── Module D: Visual Rendering ────────────────────────────────────
            if visualizer:
                visualizer.render_vector(
                    vector=vector,
                    base=base,
                    sequence_fragment=vector.sequence_fragment,
                )

    # ── Finalize Outputs ──────────────────────────────────────────────────────
    if visualizer:
        visualizer.stop()

    if synthesizer and render_audio:
        synthesizer.export(output_wav)

    summary = chaos_engine.get_summary()
    print(f"\n[GEN-GLITCH] Transduction complete.")
    print(f"  Critical Glitch events:  {summary['critical_glitch_events']}")
    print(f"  Final Chaos state:       {summary['final_state']}")
    if render_audio:
        print(f"  Audio output:            {output_wav}")


def _print_report_header(report: BioKernelReport) -> None:
    """Display a Bio-Kernel report summary before visualization begins."""
    print(f"\n  Sequence ID:   {report.sequence_id}")
    print(f"  Length:        {report.total_length:,} bp")
    print(f"  Global GC:     {report.global_gc_content:.2%}")
    print(f"  ORFs detected: {len(report.orfs)}")
    print(f"  Hotspots:      {report.hotspot_count}")
    print(f"  Max H(s):      {report.max_entropy:.6f}")
    print(f"  Mean H(s):     {report.mean_entropy:.6f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEN-GLITCH: Bio-Hacker DNA-to-Audio/Visual Transduction Engine"
    )
    parser.add_argument("--fasta",       required=True, help="Input .fasta file path")
    parser.add_argument("--output",      default=None,  help="Output .wav file path")
    parser.add_argument("--kmer",        type=int, default=100, help="k-mer window size")
    parser.add_argument("--step",        type=int, default=25,  help="k-mer step size")
    parser.add_argument("--min-orf",     type=int, default=150, help="Minimum ORF length (bp)")
    parser.add_argument("--no-audio",    action="store_true",   help="Skip audio synthesis")
    parser.add_argument("--no-visual",   action="store_true",   help="Skip visual rendering")
    parser.add_argument("--max-records", type=int, default=1,   help="Max FASTA records to process")

    args = parser.parse_args()

    transduce_sequence(
        fasta_path=args.fasta,
        output_wav=args.output,
        kmer_size=args.kmer,
        kmer_step=args.step,
        min_orf_bp=args.min_orf,
        render_audio=not args.no_audio,
        render_visual=not args.no_visual,
        max_sequences=args.max_records,
    )
