# Interactive Planarity Solver
> **Author:** [Muhammet Doğukan Bingöl]
> **Institution:** Karadeniz Technical University  
> **Course / Project:** Graph Theory


This repository presents an **interactive software tool for planarity testing and planar graph drawing**, developed as an academic project in graph theory.

The system combines a **combinatorial exact planarity test** with a **heuristic planar drawing algorithm**, allowing users to both analyze and visually explore graphs in real time.

---

## ✨ Features

- ✅ Exact planarity testing based on **rotation systems** and **backtracking**
- 🔍 Automatic detection of **non-planarity causes** using Kuratowski graph heuristics (K₅ / K₃,₃)
- 🧠 Identification of **critical edges** responsible for non-planarity
- ✏️ Interactive graph editing (add/remove vertices and edges)
- 📐 Greedy + force-based **crossing minimization drawing algorithm**
- 🌐 Web-based interface built with **Streamlit**

---

## 🧩 Method Overview

The algorithm operates in two main phases:

### 1. Exact Planarity Test
- The input graph is decomposed into connected components.
- A spanning tree is embedded first.
- Remaining edges are inserted incrementally into faces using a **rotation system**.
- A **Minimum Remaining Values (MRV)** heuristic is applied to reduce the search space.
- If embedding fails, the graph is classified as non-planar.

### 2. Planar Drawing Heuristic
- Nodes are initially placed on a circle.
- Node positions are iteratively optimized to minimize **edge crossings**.
- A hybrid approach combining **local search** and **repulsive forces** is applied.
- The algorithm stops when no further improvement is possible or crossings reach zero.

---

## 🚀 Running the Application

### Option 1: Online (Recommended)
The application is deployed online using Streamlit Cloud:

🔗 **Live Demo:**  
*(Add your Streamlit Cloud link here)*

---

### Option 2: Run Locally

#### Requirements
- Python 3.11.x or newer

#### Installation
```bash
pip install -r requirements.txt
