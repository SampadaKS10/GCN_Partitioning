# GCN Partitioning

Graph Partitioning Using Graph Convolutional Networks as described in [GAP: Generalizable Approximate Graph Partitioning Framework](https://github.com/saurabhdash/GCN_Partitioning).

## Overview

Graph partitioning is a crucial problem in many applications such as parallel computing, VLSI design, and network analysis. The GAP framework utilizes Graph Convolutional Networks (GCNs) to achieve efficient and scalable graph partitioning.

## Loss Backward Equations

To handle large graphs, the Normalized Cuts loss function is implemented using sparse Torch tensors within a custom loss class.

The Normalized Cuts loss function \( \mathbf{Z} \) is defined as:

\[ 
\mathbf{Z} = \left(\frac{\mathbf{Y}}{\Gamma}\right)(1 - \mathbf{Y})^T \circ \mathbf{A} 
\]

where \( Y_{ij} \) is the probability of node \( i \) being in partition \( j \).

\[ 
L = \sum_{\Delta_{m} \neq 0} Z_{m} 
\]

Then the gradients can be calculated by the equations:

\[ 
\frac{\partial z_{xi}}{\partial y_{ij}} = A_{i\alpha} \left( \frac{\Gamma_j (1 - y_{\alpha j}) - y_{ij} (1 - y_{ij}) D_i}{\Gamma_j^2} \right) 
\]

\[ 
\frac{\partial z_{\alpha i}}{\partial y_{ij}} = A_{i\alpha} \left( \frac{\Gamma_j (1 - y_{ij}) - y_{\alpha j} (1 - y_{\alpha j}) D_i}{\Gamma_j^2} \right) 
\]

\[ 
\frac{\partial z_{\alpha i}}{\partial y_{ij}} = A_{i\alpha} \left( \frac{(1 - y_{\alpha j}) y_{ij} D_i}{\Gamma_j^2} \right); \; i', \alpha \neq i 
\]

The Balance loss is calculated with different sums and a square, relying on autograd to compute the backloss. The loss is:

\[ 
L = \sum_{\text{reduce\_sum}} \left( \mathbf{1}^T \mathbf{Y} - \frac{n}{g} \right)^2 
\]

where \( n \) is the number of vertices and \( g \) is the number of partitions.

## Installation

### Create a virtual environment using `venv`

```bash
python3 -m venv env
```

### Source the virtual environment

```bash
source env/bin/activate
```

### Install the required packages

```bash
pip install -r requirements.txt
```

## Usage

Check the provided Jupyter notebook for detailed instructions on how to use the framework.

## Limitations

- Testing: The current implementation has only been tested on random graphs with fewer than 100 vertices.
  - A realistic benchmark for larger graphs is needed, but processing and training on large graphs is time-consuming.
  
- Balance Loss: The balance loss currently makes the model predict that vertices are equally likely for all partitions.
  - Implementing leaky ReLU didn't improve the situation. K-partitioning isn't functioning correctly without balance; it defaults to two partitions, leaving the rest empty.

- Embeddings Generation: Embeddings \( x \) are generated with PCA, but this process can be slow. For around 100 vertices, it takes about 10 minutes, with a complexity of \( O(n \log(n)) \).

- GraphSAGE: GraphSAGE has not yet been integrated into the framework. You can find more about GraphSAGE [here](https://github.com/williamleif/GraphSAGE).

