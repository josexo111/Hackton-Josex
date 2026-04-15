# Modulo para el procesamiento del FASTA y entropia

import re
import math
from typing import Generator, Iterator, NamedTuple
from dataclasses import dataclass, field
from collections import Counter


# ── Data Contracts ─────────────────────────────────────────────────────────────

@dataclass
class KmerWindow:
    """
    Represents one sliding-window analysis frame.
    Each frame carries its own entropy signature and GC-skew vector.
    """
    start: int
    end: int
    sequence: str
    gc_skew: float       # (G-C) / (G+C), range [-1.0, 1.0]
    local_entropy: float  # Shannon entropy H(s), range [0.0, 2.0]
    is_hotspot: bool = False
    hotspot_delta: float = 0.0  # deviation from species baseline GC%


@dataclass
class OpenReadingFrame:
    """
    A candidate ORF: the genomic coordinates of a start→stop codon tunnel.
    ORFs are the signal inside the noise — transduction targets.
    """
    frame: int        # reading frame: 0, 1, or 2
    start: int        # absolute position of ATG
    stop: int         # absolute position of stop codon (end-inclusive)
    length_bp: int    # length in base pairs
    sequence: str     # raw nucleotide sequence
    amino_acids: str  # translated amino acid sequence (single-letter)
    avg_entropy: float = 0.0


@dataclass
class BioKernelReport:
    # Aquí se guarda todo el reporte final del fasta
    sequence_id: str
    total_length: int
    global_gc_content: float
    kmer_windows: list[KmerWindow] = field(default_factory=list)
    orfs: list[OpenReadingFrame] = field(default_factory=list)
    hotspot_count: int = 0
    max_entropy: float = 0.0
    mean_entropy: float = 0.0


# ── Codon Translation Table (Standard Genetic Code) ───────────────────────────

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "ATG": "M",  # START
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T",
    "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "TAA": "*", "TAG": "*",  # STOP
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*",  # TGA=STOP
    "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

STOP_CODONS = {"TAA", "TAG", "TGA"}
START_CODON = "ATG"


# ── FASTA Streaming Parser ─────────────────────────────────────────────────────

def stream_fasta(filepath: str, chunk_size: int = 65_536) -> Generator[tuple[str, str], None, None]:
    # lee el fasta de a pedazos para no reventar la RAM con archivos grandes
    current_id: str = ""
    sequence_buffer: list[str] = []

    with open(filepath, "r", buffering=chunk_size) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Flush completed record before starting new one
                if current_id and sequence_buffer:
                    yield current_id, "".join(sequence_buffer).upper()
                    sequence_buffer = []

                # Extract ID: everything between '>' and first whitespace
                current_id = line[1:].split()[0]

            else:
                # Validate nucleotide characters, strip whitespace/numbers
                clean = re.sub(r"[^ATCGN]", "", line.upper())
                if clean:
                    sequence_buffer.append(clean)

    # Don't forget to flush the final record
    if current_id and sequence_buffer:
        yield current_id, "".join(sequence_buffer).upper()


def stream_sequence_chunks(
    sequence: str, window: int, step: int
) -> Iterator[tuple[int, str]]:
    """
    Sliding window iterator over a sequence string.
    Yields (start_position, kmer_substring) tuples.

    Args:
        sequence: Full nucleotide string (uppercase ATCGN).
        window:   k-mer window size (configurable, e.g. 50, 100, 500).
        step:     Stride between windows. step < window creates overlap.

    Yields:
        (int, str): Absolute start index and the k-mer subsequence.
    """
    seq_len = len(sequence)
    for i in range(0, seq_len - window + 1, step):
        yield i, sequence[i : i + window]


# ── Entropy & GC-Skew Calculators ─────────────────────────────────────────────

def calculate_shannon_entropy(kmer: str) -> float:
    """
    Compute Shannon Entropy H(s) for a nucleotide k-mer.

    Formula:
        H(s) = -∑ P_i * log₂(P_i)   for i in {A, T, C, G, N}

    Interpretation in GEN-GLITCH:
        H(s) = 0.0  → Homopolymer run (AAAAAAA). Silence.
        H(s) = 2.0  → Perfect equiprobability. Maximum complexity.
        H(s) > 1.9  → CRITICAL GLITCH threshold (triggers Chaos Engine).

    DSP Analogy:
        H(s) maps to signal bandwidth in the transduction pipeline.
        High entropy = broadband noise → sawtooth/square wave synthesis.
        Low entropy  = narrowband tone → sine wave synthesis.

    Args:
        kmer: Nucleotide substring (uppercase, ATCGN chars).

    Returns:
        float: Shannon entropy H(s) in bits, range [0.0, log₂(4) ≈ 2.0].
    """
    if not kmer:
        return 0.0

    counts = Counter(c for c in kmer if c in "ATCG")
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p_i = count / total
            entropy -= p_i * math.log2(p_i)

    return round(entropy, 6)


def calculate_gc_skew(kmer: str) -> float:
    """
    Calculate GC-Skew: directional bias between G and C content.

    Formula:
        GC-Skew = (G - C) / (G + C)

    Biological significance:
        Positive skew → G-rich strand (leading strand in replication).
        Negative skew → C-rich strand (lagging strand).
        Sharp transitions in GC-skew indicate replication origins (oriC).

    Args:
        kmer: Nucleotide substring.

    Returns:
        float: GC-Skew in range [-1.0, 1.0]. Returns 0.0 for GC-absent k-mers.
    """
    g = kmer.count("G")
    c = kmer.count("C")
    denom = g + c
    if denom == 0:
        return 0.0
    return round((g - c) / denom, 6)


def calculate_gc_content(sequence: str) -> float:
    """
    Global GC content: fraction of bases that are G or C.

    Args:
        sequence: Full or partial nucleotide string.

    Returns:
        float: GC fraction in range [0.0, 1.0].
    """
    total = len(sequence)
    if total == 0:
        return 0.0
    gc = sequence.count("G") + sequence.count("C")
    return gc / total


# ── ORF Detection Engine ───────────────────────────────────────────────────────

def find_orfs(
    sequence: str,
    min_length_bp: int = 100,
    kmer_windows: list[KmerWindow] | None = None,
) -> list[OpenReadingFrame]:
    """
    Open Reading Frame detection across all three forward reading frames.

    Algorithm:
        For each reading frame offset (0, 1, 2):
            Scan for ATG (start codon).
            Once found, scan forward in triplets for TAA/TAG/TGA (stop codons).
            Candidate ORF = [ATG ... stop codon], inclusive.
            Filter by minimum length to eliminate spurious hits.

    Frameshift Note:
        This implementation detects CANONICAL ORFs only (+1 strand, 3 frames).
        A full implementation would also search the reverse complement (-1 strand)
        and programmatic frameshifts induced by slippery sequences (XXXYYYZ motifs).
        Reverse-complement ORFs are left as a 30h extension task.

    Args:
        sequence:       Uppercase nucleotide string.
        min_length_bp:  Minimum ORF length filter. Default 100bp (~33 amino acids).
        kmer_windows:   Optional list of KmerWindow objects for entropy annotation.

    Returns:
        List of OpenReadingFrame objects, sorted by length descending.
    """
    orfs: list[OpenReadingFrame] = []

    # Build entropy lookup: position → local entropy (for ORF annotation)
    entropy_map: dict[int, float] = {}
    if kmer_windows:
        for win in kmer_windows:
            for pos in range(win.start, win.end):
                entropy_map[pos] = win.local_entropy

    for frame_offset in range(3):
        i = frame_offset
        seq_len = len(sequence)

        while i < seq_len - 2:
            codon = sequence[i : i + 3]

            if codon == START_CODON:
                # Found a start codon — scan forward for stop
                orf_start = i
                j = i + 3  # Begin reading from next codon

                while j < seq_len - 2:
                    next_codon = sequence[j : j + 3]
                    if next_codon in STOP_CODONS:
                        orf_end = j + 3
                        orf_seq = sequence[orf_start:orf_end]
                        orf_len = len(orf_seq)

                        if orf_len >= min_length_bp:
                            # Translate to amino acids
                            aa_seq = _translate_sequence(orf_seq)

                            # Compute average entropy across ORF region
                            orf_entropies = [
                                entropy_map.get(pos, 0.0)
                                for pos in range(orf_start, orf_end)
                            ]
                            avg_ent = (
                                sum(orf_entropies) / len(orf_entropies)
                                if orf_entropies else 0.0
                            )

                            orfs.append(OpenReadingFrame(
                                frame=frame_offset,
                                start=orf_start,
                                stop=orf_end,
                                length_bp=orf_len,
                                sequence=orf_seq,
                                amino_acids=aa_seq,
                                avg_entropy=round(avg_ent, 4),
                            ))
                        break
                    j += 3

                i = j + 3  # Skip past this ORF, avoid nested overlaps
            else:
                i += 3

    orfs.sort(key=lambda o: o.length_bp, reverse=True)
    return orfs


def _translate_sequence(nucleotide_seq: str) -> str:
    """
    Translate a nucleotide sequence to amino acids using the standard codon table.

    Args:
        nucleotide_seq: In-frame nucleotide string starting with ATG.

    Returns:
        str: Single-letter amino acid sequence (stops at '*').
    """
    aa_seq = []
    for i in range(0, len(nucleotide_seq) - 2, 3):
        codon = nucleotide_seq[i : i + 3]
        aa = CODON_TABLE.get(codon, "?")
        if aa == "*":
            break
        aa_seq.append(aa)
    return "".join(aa_seq)


# ── Mutation Hotspot Detection ─────────────────────────────────────────────────

def detect_mutation_hotspots(
    windows: list[KmerWindow],
    baseline_gc: float,
    hotspot_threshold: float = 0.15,
) -> list[KmerWindow]:
    """
    Flag k-mer windows as mutation hotspots when their local GC content
    deviates significantly from the species baseline.

    Biological Rationale:
        Regions with anomalous GC content experience elevated mutation rates due to:
          - Altered DNA polymerase fidelity (GC-rich hairpin structures)
          - CpG dinucleotide methylation/deamination (C→T transitions)
          - Slipped-strand mispairing in low-complexity (AT-rich) repeats

    Args:
        windows:             List of KmerWindow objects from sliding window analysis.
        baseline_gc:         Species-level global GC fraction (e.g. 0.50 for H. sapiens).
        hotspot_threshold:   Minimum absolute GC deviation to qualify as a hotspot.
                             Default 0.15 = 15 percentage point deviation.

    Returns:
        Updated list of KmerWindow objects with is_hotspot and hotspot_delta populated.
    """
    for win in windows:
        local_gc = calculate_gc_content(win.sequence)
        delta = abs(local_gc - baseline_gc)
        if delta >= hotspot_threshold:
            win.is_hotspot = True
            win.hotspot_delta = round(delta, 4)

    return windows


# ── Main Bio-Kernel Pipeline ───────────────────────────────────────────────────

def run_bio_kernel(
    filepath: str,
    kmer_size: int = 100,
    kmer_step: int = 25,
    min_orf_bp: int = 150,
    hotspot_threshold: float = 0.15,
    max_sequences: int | None = None,
) -> Generator[BioKernelReport, None, None]:
    """
    Full Bio-Kernel pipeline: ingests a .fasta file and yields one
    BioKernelReport per sequence record.

    Design for 100MB+ Files:
    ─────────────────────────
    Uses nested generators: stream_fasta yields one sequence at a time,
    and stream_sequence_chunks yields one k-mer window at a time.
    At peak, RAM holds only: current sequence + current k-mer window list.
    For a 100MB file with 50 chromosomes, each chromosome (~2MB) is processed
    independently before the next is read from disk.

    Args:
        filepath:           Path to FASTA file.
        kmer_size:          Sliding window size in bp. Affects entropy resolution.
        kmer_step:          Stride between windows. step=kmer_size/2 → 50% overlap.
        min_orf_bp:         Minimum ORF length filter.
        hotspot_threshold:  GC deviation threshold for hotspot flagging.
        max_sequences:      Optional cap on records processed (useful for previews).

    Yields:
        BioKernelReport: One per FASTA record.
    """
    records_processed = 0

    for seq_id, sequence in stream_fasta(filepath):
        if max_sequences and records_processed >= max_sequences:
            return

        global_gc = calculate_gc_content(sequence)

        # ── Step 1: Sliding Window Analysis ───────────────────────────────────
        windows: list[KmerWindow] = []
        # print(f"Analizando ventana {start_pos}")
        for start_pos, kmer in stream_sequence_chunks(sequence, kmer_size, kmer_step):
            entropy = calculate_shannon_entropy(kmer)
            gc_skew = calculate_gc_skew(kmer)

            windows.append(KmerWindow(
                start=start_pos,
                end=start_pos + len(kmer),
                sequence=kmer,
                gc_skew=gc_skew,
                local_entropy=entropy,
            ))

        # ── Step 2: Mutation Hotspot Detection ────────────────────────────────
        windows = detect_mutation_hotspots(windows, global_gc, hotspot_threshold)
        hotspot_count = sum(1 for w in windows if w.is_hotspot)

        # ── Step 3: ORF Detection ─────────────────────────────────────────────
        orfs = find_orfs(sequence, min_orf_bp, windows)

        # ── Step 4: Aggregate Stats ───────────────────────────────────────────
        entropies = [w.local_entropy for w in windows]
        max_entropy = max(entropies) if entropies else 0.0
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        yield BioKernelReport(
            sequence_id=seq_id,
            total_length=len(sequence),
            global_gc_content=round(global_gc, 4),
            kmer_windows=windows,
            orfs=orfs,
            hotspot_count=hotspot_count,
            max_entropy=round(max_entropy, 6),
            mean_entropy=round(mean_entropy, 6),
        )

        records_processed += 1