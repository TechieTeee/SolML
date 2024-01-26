# SolML
Rust machine learning script for Solana data to help increase adoption of Rust with machine learning

## Table of Contents
- [Introduction](#introduction)
- [Purpose](#purpose)
- [Tech Stack](#tech-stack)
- [Applications for Real World Use Cases](#applications-for-real-world-use-cases)
- [How it Can Help Developers](#how-it-can-help-developers)
- [How to Run the Script](#how-to-run-the-script)
- [Contributing](#contributing)

## Introduction
SolML is a machine learning project that leverages Solana blockchain data for analysis and model training. It applies machine learning to perform various algorithms on Solana data.

## Purpose
The project aims to provide insights into Solana blockchain data using machine learning techniques. It explores wealth concentration, decentralization, security, and performance aspects of the Solana network and make it easier for Solana/Rust developers to develop and run machine learning models.

## Tech Stack
- Rust
- Solana Client
- Solana SDK
- Quicknode Solana API
- Linfa
- Isahc

## Current Algorithms Offered
- Decision Tree
- Logistic Regression
- K-Means

## Applications for Real World Use Cases
SolML can be used to analyze and understand various aspects of the Solana blockchain. Potential real-world use cases include:
- Assessing wealth concentration in the network
- Analyzing decentralization through cluster nodes and slot leaders
- Evaluating network health, security, and performance

## How it Can Help Developers
Developers can benefit from SolML in the following ways:
- Gain insights into Solana blockchain data using machine learning
- Understand patterns in wealth distribution, decentralization, and network health
- Use machine learning models for predictive analysis based on Solana data

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
If you'd like to contribute to SolML, please feel free to do so.
