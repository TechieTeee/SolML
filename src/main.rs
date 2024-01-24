use dotenv::dotenv;
use isahc;
use linfa::prelude::*;
use solana_client::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::signature::Signature;
use std::env;
use std::error::Error;

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
    let balance = client.get_balance(&Signature::new(&[0; 64])).await?;
    let largest_accounts = client.get_token_largest_accounts("SOL".to_string(), 10).await?;
    println!("Wealth Concentration:");
    println!("- Overall balance: {}", balance);
    println!("- Largest SOL accounts: {:?}", largest_accounts);

    // Properly close the async block
    Ok(())
}

async fn fetch_solana_data(endpoint: &str) -> Result<Vec<SolanaData>, isahc::Error> {
    let response = isahc::get(endpoint)?.text()?;
    let solana_data: Vec<SolanaData> = serde_json::from_str(&response)?;
    Ok(solana_data)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Retrieve the QuickNode endpoint from the environment variable
    let quicknode_endpoint = env::var("QUICKNODE_ENDPOINT").expect("QUICKNODE_ENDPOINT not set in .env file");

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

    // Extract labels based on your analysis goals
    let labels: Vec<f64> = solana_data.iter().map(|data| data.balance as f64).collect();

    let dataset = Dataset::new(features, labels);

    // Apply Linfa algorithms
    // 1. Reduction using PCA
    let pca_model = linfa::pca().fit(&dataset).unwrap();
    let reduced_data = pca_model.transform(&dataset).unwrap();
    println!("Reduced Data: {:?}", reduced_data);

    // 2. Logistic Regression
    let logistic_regression_model = linfa::logistic_regression().fit(&dataset).unwrap();
    let prediction_lr = logistic_regression_model.predict(&dataset.records());
    println!("Logistic Regression Prediction: {:?}", prediction_lr);

    // 3. K-Means clustering
    let kmeans_model = linfa::kmeans(3).fit(&dataset).unwrap();
    let labels_kmeans = kmeans_model.predict(&dataset.records());
    println!("K-Means Labels: {:?}", labels_kmeans);

    // Establish connection to Solana RPC node
    let solana_rpc_url = "https://api.mainnet-beta.solana.com".to_string();
    let solana_client = RpcClient::new(solana_rpc_url);

    // Assess Solana health
    assess_solana_health(&solana_client).await?;

    Ok(())
}
