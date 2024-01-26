# SolML
Rust machine learning script for Solana data to facilitate the integration of Rust with machine learning.

## Table of Contents
- [Introduction](#introduction)
- [Purpose](#purpose)
- [Tech Stack](#tech-stack)
- [Available Algorithms](#available-algorithms)
- [Applications for Real-World Use Cases](#applications-for-real-world-use-cases)
- [How it Can Help Developers](#how-it-can-help-developers)
- [How to Run the Script](#how-to-run-the-script)
- [Contributing](#contributing)

## Introduction
SolML is a Rust-based machine learning project designed for analyzing Solana blockchain data. It leverages various machine learning algorithms to provide insights and patterns from the Solana network.

## Purpose
The primary goal of this project is to empower Solana and Rust developers with tools to explore and understand Solana blockchain data through machine learning. SolML focuses on aspects such as wealth concentration, decentralization, security, and network performance.

## Tech Stack
- Rust
- Solana Client
- Solana SDK
- Quicknode Solana API
- Linfa
- Isahc

## Available Algorithms
### 1. Decision Tree
Decision Trees are used for predictive modeling. They work well for classification tasks and provide a clear visualization of decision-making processes.

### 2. Logistic Regression
Logistic Regression is commonly used for binary classification tasks. It's effective for understanding the relationship between independent and dependent variables.

### 3. K-Means Clustering
K-Means is an unsupervised learning algorithm used for clustering. It's beneficial for identifying patterns and grouping similar data points.

## Applications for Real-World Use Cases
SolML can be applied to various real-world scenarios, including:
- Assessing wealth distribution within the Solana network
- Analyzing decentralization through cluster nodes and slot leaders
- Evaluating network health, security, and performance metrics

## How it Can Help Developers
Developers can utilize SolML to:
- Gain valuable insights from Solana blockchain data through machine learning techniques
- Understand patterns related to wealth distribution, decentralization, and network health
- Employ machine learning models for predictive analysis based on Solana data

## How to Run the Script
### Prerequisites
- Rust installed
- .env file with the QuickNode endpoint (`QUICKNODE_ENDPOINT=https://your-quicknode-endpoint`)

### Setup
1. Clone the repository: `git clone https://github.com/TechieTeee/SolML.git`
2. Navigate to the project directory: `cd SolML`

### Running the Script
1. Set up the environment variables in a `.env` file.
2. Run the script: `cargo run`

## Contributing
Contributions to SolML are welcome! Feel free to contribute by forking the repository and creating pull requests.
