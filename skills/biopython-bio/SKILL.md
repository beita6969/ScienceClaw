---
name: biopython-bio
description: "Bioinformatics operations via Biopython. Use when: user asks about DNA/protein sequences, BLAST, or PDB structures. NOT for: clinical genomics or variant calling pipelines."
metadata: { "openclaw": { "emoji": "\uD83E\uDDEC", "requires": { "bins": ["python3"] }, "install": [{ "id": "uv-biopython", "kind": "uv", "package": "biopython" }] } }
---

# Biopython Bio

Bioinformatics operations using Biopython.

## When to Use

- Reading/writing sequence files (FASTA, GenBank)
- Running BLAST searches (local or remote NCBI)
- Sequence alignment and manipulation
- Parsing PDB protein structures
- Phylogenetic tree construction
- Querying NCBI Entrez databases

## When NOT to Use

- Clinical genomics or variant calling (use GATK, bcftools)
- RNA-seq differential expression (use DESeq2, edgeR)
- Genome assembly (use SPAdes, Canu)
- Molecular dynamics simulations (use GROMACS, OpenMM)

## Sequence Reading and Writing

```python
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

for record in SeqIO.parse('sequences.fasta', 'fasta'):
    print(f"{record.id}: {len(record.seq)} bp")

# Write FASTA
records = [SeqRecord(Seq('ATGCGATCGATCG'), id='seq1', description='example')]
SeqIO.write(records, 'output.fasta', 'fasta')
```

## Sequence Manipulation

```python
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, molecular_weight

dna = Seq('ATGCGATCGATCGATCG')
rev_comp = dna.reverse_complement()
protein = dna.translate()
gc = gc_fraction(dna)
mw = molecular_weight(dna, seq_type='DNA')
```

## BLAST Searches

```python
from Bio.Blast import NCBIWWW, NCBIXML

result_handle = NCBIWWW.qblast('blastn', 'nt', 'ATGCGATCGATCGATCG')
for record in NCBIXML.parse(result_handle):
    for aln in record.alignments:
        for hsp in aln.hsps:
            if hsp.expect < 1e-10:
                print(f"{aln.title[:60]}, E={hsp.expect}")
```

## Pairwise Alignment

```python
from Bio import Align

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 2
aligner.mismatch_score = -1
best = aligner.align('ATCGATCGATCG', 'ATCAATCAATCG')[0]
print(best, f"Score: {best.score}")
```

## PDB Structure Parsing

```python
from Bio.PDB import PDBParser, PDBList

structure = PDBParser(QUIET=True).get_structure('prot', 'structure.pdb')
for chain in structure[0]:
    for res in chain:
        if res.id[0] == ' ' and 'CA' in res:
            print(f"{res.resname} {res.id[1]}: {res['CA'].coord}")
```

## Entrez Queries and Phylogenetics

```python
from Bio import Entrez, Phylo, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

Entrez.email = 'your.email@example.com'  # Required by NCBI
handle = Entrez.esearch(db='pubmed', term='CRISPR AND 2025[pdat]', retmax=5)
record = Entrez.read(handle)

# Phylogenetics from alignment
aln = AlignIO.read('aligned.fasta', 'fasta')
tree = DistanceTreeConstructor().nj(DistanceCalculator('identity').get_distance(aln))
Phylo.draw_ascii(tree)
```

## Quick One-liner

```bash
python3 -c "
from Bio.Seq import Seq
dna = Seq('ATGAAAGCTTGA')
print(f'Protein: {dna.translate()}, RevComp: {dna.reverse_complement()}')
"
```

## Best Practices

1. Always set `Entrez.email` before NCBI queries.
2. Respect NCBI rate limits: max 3 requests/second without API key.
3. Use `QUIET=True` in PDB parser to suppress warnings.
4. Check sequence type before operations like `translate()`.
5. For large BLAST jobs, prefer local BLAST+ over remote NCBI.
