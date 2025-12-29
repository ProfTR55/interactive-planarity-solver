# Interactive Planarity Solver

> **Author:** Muhammet Doğukan Bingöl  
> **Institution:** Karadeniz Technical University  
> **Course / Project:** Graph Theory  

---

## 📌 Overview

This repository presents an **interactive software tool for planarity testing and planar graph visualization**, developed as an academic project in graph theory.

The system integrates an **exact combinatorial planarity testing algorithm** with a **heuristic drawing method** aimed at reducing edge crossings.  
Users can construct graphs interactively, test their planarity, analyze the causes of non-planarity, and generate intuitive planar drawings when possible.

The primary goal of this project is to provide an **explainable and educational planarity solver**, rather than a black-box decision tool.

---

## ✨ Key Features

- ✅ **Exact planarity testing** using rotation systems and backtracking  
- 🔍 **Heuristic detection of non-planarity causes** via Kuratowski subgraph patterns (K₅ and K₃,₃)  
- 🧠 **Critical edge identification** to explain and resolve non-planarity  
- ✏️ Fully **interactive graph editing** (add/remove vertices and edges)  
- 📐 **Greedy + force-based planar drawing heuristic** focused on minimizing edge crossings  
- 🌐 Web-based interactive interface implemented with **Streamlit**

---

## 🧩 Methodology Overview

The system operates in two main algorithmic phases:

---

### 1️⃣ Exact Planarity Testing

- The input graph is first decomposed into **connected components**.
- Each component is tested independently for planarity.
- A **spanning tree** is embedded as an initial planar structure.
- Remaining edges are incrementally inserted into faces using a **rotation system representation**.
- A **Minimum Remaining Values (MRV)** heuristic selects the next edge to be embedded, significantly reducing the backtracking search space.
- If no valid embedding is found, the graph is classified as **non-planar**.

To improve interpretability, the algorithm attempts to identify:
- Kuratowski-type obstructions (K₅ or K₃,₃), and  
- edges that most frequently cause embedding failure across randomized trials (critical edges).

---

### 2️⃣ Planar Drawing Heuristic

- Vertices are initially placed on a circular layout with small random perturbations.
- Node positions are iteratively optimized to minimize the **number of edge crossings**.
- For each vertex, multiple candidate positions are evaluated using a **local greedy search**.
- A global **repulsive force step** is applied between iterations to improve spacing and readability.
- The process terminates when no further improvement is possible or when a crossing-free drawing is obtained.

This drawing approach is independent of the planarity test and is designed to be efficient and visually intuitive.

---

## 🚀 Running the Application

### Option 1: Online (Recommended)

The application is deployed using **Streamlit Cloud** and can be accessed directly via a web browser:

🔗 **Live Demo:**  
*(Add your Streamlit Cloud link here)*

> Note: If the application has not been accessed for a long time, initial loading may take a few seconds due to cold start behavior.

---

### Option 2: Run Locally

#### Requirements
- Python **3.11.x** or newer

#### Installation
```bash
pip install -r requirements.txt
