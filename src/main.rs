use dotenv::dotenv;
use linfa::prelude::*;
use linfa_clustering::{KMeans};
use linfa_logistic::LogisticRegression;
use linfa_trees::DecisionTree;
use solana_client::rpc_client::RpcClient;
use std::env;
use std::error::Error;
use serde::Deserialize;
use isahc::ReadResponseExt;
use solana_sdk::pubkey::Pubkey;
use ndarray::{arr1, Array2, stack, Axis, ArrayView1};
use linfa::traits::Fit;


#[derive(Debug, Deserialize)]
struct SolanaData {
    balance: u64,
    largest_accounts: Vec<u64>,
    cluster_nodes: usize,
    slot_leaders: Vec<String>,
    health_status: String,
    block_production_rate: u64,
}

async fn assess_solana_health(client: &RpcClient) -> Result<(), Box<dyn Error>> {
    // 1. Wealth Concentration Assessment
    let balance = client.get_balance(&Pubkey::new(&[0u8; 64]))?;
    let largest_accounts = client.get_token_largest_accounts(&Pubkey::new(&[0u8; 64]))?;
    println!("Wealth Concentration:");
    println!("- Overall balance: {}", balance);
    println!("- Largest SOL accounts: {:?}", largest_accounts);

    Ok(())
}

async fn fetch_solana_data(endpoint: &str) -> Result<Vec<SolanaData>, isahc::Error> {
    let response = isahc::get(endpoint)?.text()?;
    let solana_data: Vec<SolanaData> = serde_json::from_str(&response).expect("Failed to parse JSON");
    Ok(solana_data)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Retrieve the QuickNode endpoint from the environment variable
    let quicknode_endpoint =
        env::var("QUICKNODE_ENDPOINT").expect("QUICKNODE_ENDPOINT not set in .env file");

    // Fetch Solana data
    // Choose what Solana, data endpoints you want to use from Quicknode
    // Check out the endpoints available in the Quicknode documentation
    let solana_data: Vec<SolanaData> = fetch_solana_data(&quicknode_endpoint).await?;

    // Convert SolanaData to a linfa dataset
    let features: Vec<f64> = solana_data
        .iter()
        .map(|data| vec![
            data.balance as f64,
            data.largest_accounts.len() as f64,
            data.cluster_nodes as f64,
        ])
        .flatten()
        .collect();

    // Create features_array
    let mut features_array = Vec::new();
    for chunk in features.chunks(3) {
        features_array.push(arr1(chunk));
    }

    // Convert to Vec<ArrayView1<f64>>
    let features_array: Vec<ArrayView1<f64>> = features_array.iter().map(|a| a.view()).collect();

    // Stack arrays along Axis(0)
    let features_array: Array2<f64> =
        stack(Axis(0), features_array.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice()).unwrap();

    // Extract labels based on your analysis goals
    let labels: Vec<usize> = solana_data.iter().map(|data| data.balance as usize).collect();

    // Create Dataset
    let dataset = Dataset::new(features_array.clone(), arr1(&labels));

    // Apply Linfa algorithms based on user preferences
     // Set to true or false based on whether you want to use this algorithm
    let enable_decision_tree = true;
    if enable_decision_tree {
        let decision_tree_model = DecisionTree::params().fit(&dataset)?;
        println!("Decision Tree Prediction: {:?}", decision_tree_model);
    }

     // Set to true or false based on whether you want to use this algorithm
    let enable_logistic_regression = true;
    if enable_logistic_regression {
        let logistic_regression_model = LogisticRegression::default()
            .max_iterations(150)
            .fit(&dataset)?;
        println!("Logistic Regression Prediction: {:?}", logistic_regression_model);
    }

     // Set to true or false based on whether you want to use this algorithm
    let enable_kmeans = true; 
    if enable_kmeans {
        let kmeans_model = KMeans::params(3).fit(&dataset)?;
        println!("K-Means Labels: {:?}", kmeans_model);
    }

    // Establish connection to Solana RPC node
    let solana_rpc_url = "https://api.mainnet-beta.solana.com".to_string();
    let solana_client = RpcClient::new(solana_rpc_url);

    // Assess Solana health
    assess_solana_health(&solana_client).await?;

    Ok(())
}
