use dotenv::dotenv;
use linfa::prelude::*;
use linfa_reduction::Pca;
use linfa_clustering::{KMeans};
use linfa_logistic::LogisticRegression;
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
    let labels: Vec<f64> = solana_data.iter().map(|data| data.balance as f64).collect();

    // Create dataset
    let dataset = Dataset::new(features_array.clone(), arr1(&labels));

    // Apply Linfa algorithms
    // 1. Reduction using PCA (generic wrapper for open-ended output type)
    fn my_pca<T: linfa::Float>(dataset: &Dataset<T>) -> Result<Pca<T>, Box<dyn Error>> {
        Pca::<T>::fit::<_, _, _>(dataset)
    }

    let pca_model = my_pca(&dataset)?; // Use the generic wrapper
    let reduced_data = pca_model.transform(&dataset)?;
    println!("Reduced Data: {:?}", reduced_data);

    // 2. Logistic Regression (using predict on the model instance)
    let logistic_regression_model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&dataset)?;
    let prediction_lr = logistic_regression_model.predict(&dataset.records())?;
    println!("Logistic Regression Prediction: {:?}", prediction_lr);

    // 3. K-Means clustering (using the new constructor)
    let kmeans_model = KMeans::new(3).fit(&dataset)?;
    let labels_kmeans = kmeans_model.predict(&dataset.records());
    println!("K-Means Labels: {:?}", labels_kmeans);
    

    // Establish connection to Solana RPC node
    let solana_rpc_url = "https://api.mainnet-beta.solana.com".to_string();
    let solana_client = RpcClient::new(solana_rpc_url);

    // Assess Solana health
    assess_solana_health(&solana_client).await?;

    Ok(())
}
