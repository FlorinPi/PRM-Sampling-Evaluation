# PRM Sampling Evaluation
## Overview
This repository contains a modular framework for experimenting with Probabilistic Roadmap (PRM) path planning in 2D environments. The project evaluates and compares different PRM sampling strategies and studies how key hyperparameters influence roadmap quality, planning success, and computation time.

Four sampling strategies are implemented and tested—random, grid-based, Gaussian, and bridge sampling—across four custom-designed 2D map scenarios. For each configuration, the framework runs repeated experiments, collects performance statistics, and generates visualizations of the best-performing solutions.

The code is designed to be easily extensible, allowing new sampling strategies, custom maps, and experimental settings to be added with minimal changes.

## Project Scope
The framework evaluates PRM performance across different scenarios by measuring:
* Roadmap construction time
* Path length between start and goal
* Success rate across multiple runs

In addition to comparing sampling strategies, the project systematically varies key hyperparameters—such as the number of samples, number of neighbors, and sampling bias parameters—to analyze their influence on performance. Each hyperparameter is swept independently while others are held at nominal values, enabling clear insight into under- and over-parameterization effects.

For each experiment, the best-performing run (shortest feasible path) is identified and visualized.

## Code Overview

`PRM.py` Contains the core PRM implementation. This includes:
* Sampling methods (random, grid, Gaussian, bridge)
* Roadmap construction using nearest-neighbor connections
* Collision checking
* Shortest-path search using Dijkstra’s algorithm
* Visualization of roadmaps and paths

Sampling strategies are implemented as modular methods, making it straightforward to add new ones.

`main.py` Handles experiment setup and evaluation. This script:
* Loads scenario maps
* Runs repeated trials for each sampling strategy
* Sweeps hyperparameters
* Collects statistics (success rate, path length, build time)
* Saves result summaries and visualizations

Scenarios and parameters can be easily extended or modified here.

## Setup
Python version: 3.13.11

Install the required dependencies using:
```
pip install numpy
pip install matplotlib
pip install networkx
pip install scipy
```
Once the dependencies are installed, experiments can be run directly from `main.py`
