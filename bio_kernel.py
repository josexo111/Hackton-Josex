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


