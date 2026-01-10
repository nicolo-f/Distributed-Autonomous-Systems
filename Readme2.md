# Distributed Autonomous Systems - Course Project (2025/2026)

**University of Bologna** **Course:** Distributed Autonomous Systems  
**Professors:** Giuseppe Notarstefano, Ivano Notarnicola  

**Group Members:**
* Nicolo Faedo
* Mirco Paltrinieri
* Luca Sabatini

---

## Project Overview

This repository contains the final exam project for the Distributed Autonomous Systems course. The work focuses on the implementation and evaluation of distributed optimization strategies for multi-robot fleets, specifically addressing:

1.  **Multi-Robot Target Localization (Task 1):** Application of the Gradient Tracking algorithm for cooperative estimation of target positions.
2.  **Aggregative Fleet Control (Task 2):** Implementation of an Aggregative Tracking algorithm with collision avoidance to manage fleet cohesion and target tracking.

The project includes theoretical validation via Python simulations and a practical distributed implementation using ROS 2.

---

## Folder Structure

The submission is organized as follows:

* **`report_group_XX.pdf`** The complete project report containing the theoretical background, algorithm descriptions, and detailed analysis of the experimental results.

* **`report/`** Contains the LaTeX source code and figure assets used to generate the PDF report.
    * `report_group_XX.tex`: Main LaTeX file.
    * `figs/`: Folder containing plots, diagrams, and simulation screenshots.

* **`code/`** Contains all the software developed for this project.
    * **Python Simulations:** Standalone scripts (`Task1.py`, `Task2.1.py`) for validating the algorithms in a centralized environment.
    * **ROS 2 Package:** The `task2` package containing the distributed implementation (nodes, launch files, and visualizers) for the Aggregative Fleet Control task.
    * **`README.md`**: Detailed instructions on how to install dependencies and run the simulations.

---

## Quick Start

To view the project results, please refer to **`report_group_XX.pdf`**.

To run the code:
1.  Navigate to the `code/` folder.
2.  Follow the instructions in the `code/README.md` file to set up the Python environment or build the ROS 2 package.