# Interactive Planarity Solver

> **Author:** Muhammet Doğukan Bingöl  
> **Institution:** Karadeniz Technical University  
> **Course / Project:** Graph Theory  

---

## Abstract

This repository presents an **interactive software system for exact planarity testing and planar graph drawing**, developed as part of an academic study in graph theory.

The system combines an **exact combinatorial planarity testing algorithm**, based on rotation systems and backtracking, with a **heuristic drawing framework** aimed at producing readable planar (or near-planar) embeddings.  
Beyond deciding planarity, the implementation emphasizes **explainability**, offering structural insights into the sources of non-planarity.

### Online Access (Recommended)

The application is deployed via **Streamlit Cloud** and can be accessed at:

🔗 **Live Demo**  
https://interactive-planarity-solver-dogukanv2.streamlit.app
> The application may experience a brief delay on first access due to cold-start behavior of the hosting platform.


---

## Key Contributions

- An **exact planarity testing algorithm** based on rotation systems and face-preserving edge insertion  
- A **search-space reduction strategy** using a Minimum Remaining Values (MRV) heuristic  
- Heuristic detection of **Kuratowski-type obstructions** (K₅ and K₃,₃)  
- A randomized analysis framework for identifying **critical edges** responsible for non-planarity  
- A **hybrid greedy and force-based drawing algorithm** for graph visualization  
- A fully **interactive web-based interface** enabling real-time graph manipulation  

---

## Methodological Overview

The system consists of two main algorithmic components.

---

### 1. Exact Planarity Testing

Planarity is decided using a constructive embedding approach:

- The input graph is decomposed into its **connected components**, which are processed independently.
- A **spanning tree** is embedded first to establish an initial planar structure.
- Remaining edges are incrementally inserted into existing faces using a **rotation system representation**.
- At each step, the next edge to be embedded is selected via a **Minimum Remaining Values (MRV)** heuristic, prioritizing edges with the fewest feasible insertion options.
- The embedding process is performed using **backtracking**; failure to embed all edges implies non-planarity.

To improve interpretability, additional heuristics are applied to:
- detect Kuratowski-type substructures, and  
- identify edges that most frequently cause embedding failure across multiple randomized trials.

---

### 2. Planar Drawing Heuristic

For visualization, an independent drawing heuristic is employed:

- Vertices are initially placed on a circular layout with small random perturbations to break symmetry.
- Vertex positions are iteratively refined to minimize the **number of edge crossings**.
- For each vertex, multiple candidate positions are evaluated using a **local greedy search strategy**.
- Between iterations, a **global repulsive force** is applied to improve vertex distribution and readability.
- The algorithm terminates when no further improvement is observed or when a crossing-free drawing is obtained.

This drawing procedure is independent of the planarity test and is designed to be computationally efficient while producing intuitive layouts.


### Local Execution

#### Requirements
- Python **3.11.x** or newer

#### Installation
```bash
pip install -r requirements.txt
