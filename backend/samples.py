"""
Pre-bundled sample documents for instant testing without needing to upload files.
"""

SAMPLE_DOCUMENTS = {
    "quantum_computing": {
        "title": "Quantum Computing & Qubit Architecture.txt",
        "description": "A technical primer on quantum superposition, entanglement, and quantum error correction.",
        "content": """# Introduction to Quantum Computing and Qubit Architectures

Quantum computing represents a fundamental shift in computation by leveraging the principles of quantum mechanics. Unlike classical computers, which process information using binary bits that exist strictly as 0 or 1, quantum computers utilize quantum bits, or qubits.

## Superposition and Entanglement
A qubit can exist in a linear combination of states |0⟩ and |1⟩ simultaneously, a phenomenon known as superposition:
|ψ⟩ = α|0⟩ + β|1⟩
where α and β are complex probability amplitudes such that |α|² + |β|² = 1.

Entanglement is a unique quantum phenomenon where multiple qubits become correlated such that the quantum state of each qubit cannot be described independently of the state of the others. Measuring one entangled qubit instantaneously determines the state of its paired partner, regardless of spatial separation.

## Physical Qubit Implementations
Several physical modalities are actively being engineered for fault-tolerant quantum computing:

1. Superconducting Circuits:
Employed by IBM and Google, superconducting transmon qubits utilize Josephson junctions operating at millikelvin temperatures (~15 mK) inside dilution refrigerators. They offer nanosecond gate speeds but suffer from relatively short coherence times (T1 and T2 ~ 100-300 microseconds).

2. Trapped-Ion Systems:
Employed by IonQ and Quantinuum, trapped-ion systems suspend charged atomic ions (such as Ytterbium-171 or Barium-133) in electromagnetic fields using radiofrequency Paul traps. They boast record gate fidelities (>99.9%) and coherence times lasting minutes, though two-qubit gate operations are significantly slower (~10-100 microseconds).

3. Neutral Atom Arrays:
Neutral atoms trapped in optical tweezers manipulated by laser beams provide high scalability and reconfigurable geometric topologies in 2D and 3D architectures.

4. Topological Qubits:
Theoretical qubits utilizing non-Abelian anyons (Majorana zero modes) offer hardware-level fault tolerance against environmental decoherence, pursued intensively by Microsoft.

## Quantum Error Correction (QEC)
Current quantum devices operate in the Noisy Intermediate-Scale Quantum (NISQ) era. Due to environmental thermal noise, magnetic fluctuations, and cosmic rays, physical error rates hover around 10⁻³ per gate. Fault-tolerant algorithms, such as Shor's algorithm for factoring 2048-bit RSA keys, require error rates below 10⁻¹⁰.

The Surface Code is the leading topological error-correcting code. It requires approximately 1,000 to 10,000 physical qubits to construct a single fault-tolerant logical qubit with a distance d=27 lattice. Syndrome measurement rounds detect bit-flip (X) and phase-flip (Z) errors without collapsing the underlying logical quantum state.
"""
    },
    "enterprise_ai": {
        "title": "Enterprise RAG Architecture & Vector Search.txt",
        "description": "Technical specification for production Retrieval-Augmented Generation systems.",
        "content": """# Enterprise Architecture for Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) bridges non-parametric enterprise knowledge bases with parametric Large Language Models (LLMs) to eliminate hallucinations, inject domain-specific context, and enforce strict data access controls.

## 1. Document Ingestion and Chunking Strategies
The ingestion pipeline processes unstructured files (PDF, DOCX, Markdown, HTML) through document layout analyzers:
- Semantic Chunking: Splits text at natural discourse boundaries (headings, paragraphs, semantic sentence shifts) rather than arbitrary token boundaries. Typical chunk sizes range from 256 to 512 tokens with a 10-20% overlap.
- Overlap Preservation: Overlap ensures contiguous context across chunk boundaries, preventing fragmented entity definitions and broken multi-sentence reasoning chains.
- Metadata Enrichment: Every chunk is tagged with document source, author, section hierarchy, page numbers, timestamp, and security classification tags.

## 2. Embedding Generation and Dimensionality
Embedding models project text tokens into a continuous vector space where semantic proximity corresponds to geometric distance:
- Dense Embeddings: High-dimensional representations (e.g., 768 or 1536 dimensions) capturing conceptual nuances, synonyms, and cross-lingual semantics.
- Distance Metrics: Cosine Similarity and Inner Dot Product are primarily used for normalized vectors. Cosine similarity calculates cos(θ) = (A · B) / (||A|| ||B||), returning values from -1.0 to +1.0.
- Sparse Embeddings (BM25): Token-matching frequencies provide exact keyword matching for serial numbers, acronyms, and product identifiers.
- Hybrid Search: Combines dense vector retrieval with sparse BM25 retrieval using Reciprocal Rank Fusion (RRF) to maximize recall and precision.

## 3. Vector Database Indexing
Vector databases employ Approximate Nearest Neighbor (ANN) index algorithms:
- HNSW (Hierarchical Navigable Small World): A multi-layered graph structure offering sub-millisecond search latencies and high recall (>98%).
- IVF-PQ (Inverted File with Product Quantization): Clusters vectors into Voronoi cells and quantizes vectors to reduce RAM consumption by up to 80%.

## 4. Re-ranking and Context Filtering
Top-k chunks (e.g., k=20) retrieved by the vector index are fed through a cross-encoder re-ranker model. The re-ranker computes full cross-attention between the query and candidate passages, filtering out irrelevant chunks before the top 3-5 passages are passed into the LLM context window.
"""
    },
    "clean_energy": {
        "title": "Grid-Scale Energy Storage & Lithium-Iron-Phosphate.txt",
        "description": "Analysis of renewable energy storage systems, battery chemistries, and grid stability.",
        "content": """# Grid-Scale Energy Storage Systems and Battery Chemistries

The global transition to intermittent renewable energy sources, specifically solar photovoltaics and wind turbines, requires massive deployment of Battery Energy Storage Systems (BESS) to balance electricity supply and demand, provide frequency regulation, and prevent curtailment.

## Battery Chemistries Comparison

1. Lithium Iron Phosphate (LFP - LiFePO4):
LFP has become the dominant chemistry for stationary energy storage systems, representing over 85% of new utility-scale installations.
- Key advantages: Superior thermal runaway threshold (~270°C vs 210°C for NMC), cobalt-free and nickel-free composition reducing ethical supply chain risks, and exceptional cycle life exceeding 6,000 to 10,000 cycles at 80% Depth of Discharge (DoD).
- Limitations: Lower volumetric energy density (~300-400 Wh/L) compared to NMC, which is acceptable for stationary land-based storage.

2. Nickel Manganese Cobalt (NMC - LiNiMnCoO2):
Traditionally dominant in electric vehicles due to high energy density (up to 700 Wh/L). However, utility adoption has dropped due to elevated thermal runaway risk and higher raw material costs.

3. Sodium-Ion (Na-Ion):
Emerging alternative using abundant sodium precursors. Operates across extreme temperature windows (-20°C to +60°C) with zero risk of thermal propagation during zero-volt discharge, though energy densities remain around 140-160 Wh/kg.

4. Vanadium Redox Flow Batteries (VRFB):
Flow batteries decouple power (kW) from energy (kWh) using liquid electrolyte stored in external tanks. They offer virtually unlimited calendar lifespan (>20 years) with zero degradation, ideal for long-duration energy storage (LDES) (>8 hours).

## Power Conversion Systems (PCS) and Inverters
Grid-scale BESS connects to high-voltage transmission grids through bi-directional inverters. Grid-forming inverters (GFM) emulate synchronous machine inertia using virtual synchronous generator (VSG) algorithms, providing black-start capability and system strength in grids with low physical rotational inertia.
"""
    }
}
