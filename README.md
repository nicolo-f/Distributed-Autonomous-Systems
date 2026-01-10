# Distributed Autonomous Systems - Course Project

**University of Bologna** **Course:** Distributed Autonomous Systems (2025/2026)  
**Professors:** Giuseppe Notarstefano, Ivano Notarnicola  

**Authors:** 
* Nicolò Faedo
* Mirco Paltrinieri
* Luca Sabatini

---

## Project Overview

This repository contains the implementation of distributed optimization strategies for multi-robot fleets, developed for the Distributed Autonomous Systems course. The project is divided into two main parts:

1.  **Multi-Robot Target Localization (Task 1):** A distributed Gradient Tracking algorithm to estimate the position of targets using a fleet of sensor-equipped robots.
2.  **Aggregative Fleet Control (Task 2):** An Aggregative Tracking algorithm enabling a fleet to track private targets while maintaining cohesion (formation control) and avoiding collisions.

The code includes both standalone Python simulations for algorithm validation and a complete ROS 2 package for the distributed implementation of Task 2.

---

## Repository Structure

```text
code/
├── Task1.py            # Simulation for Task 1.2 (Target Localization)
├── Task2.1.py          # Simulation for Task 2.1 (Aggregative Tracking)
├── helper.py           # Shared helper classes (Graph generation, Cost functions)
├── drone.png           # Asset for visualization
└── task2/              # ROS 2 Package for Task 2.2
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── task2/          # Source code for ROS nodes
    │   ├── the_agent.py        # Distributed agent logic
    │   ├── task2_plotter.py    # Real-time data plotting
    │   ├── task2_visualizer.py # Visualization publisher
    │   └── helper.py           # Helper classes for ROS nodes
    ├── launch_folder/
    │   └── task2_launch.py     # Launch file for the multi-robot system
    ├── resource/       # RViz configuration and mesh files
    └── test/           # Unit tests