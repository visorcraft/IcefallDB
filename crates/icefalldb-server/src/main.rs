use clap::Parser;
use icefalldb_server::Server;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "icefalldb-server")]
struct Args {
    /// Database directory (contains one or more table subdirectories).
    #[arg(short, long)]
    db: PathBuf,
    /// TCP port to listen on.
    #[arg(short, long, default_value = "8080")]
    port: u16,
    /// Host to bind to.
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,
    /// Result-cache budget in MiB stored under `<db>/_query_cache` (0 = disabled).
    #[arg(long, default_value = "1024")]
    result_cache_mb: u64,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    match Server::new_with_cache_mb(&args.db, args.result_cache_mb).await {
        Ok(server) => {
            let addr = format!("{}:{}", args.host, args.port);
            eprintln!("icefalldb-server listening on http://{}", addr);
            if let Err(e) = server.serve(&addr).await {
                eprintln!("server error: {:#}", e);
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to start server: {:#}", e);
            ExitCode::from(1)
        }
    }
}
