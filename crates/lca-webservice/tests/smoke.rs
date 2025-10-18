#[tokio::test]
async fn health_endpoint_works() {
    let app = lca_webservice::app::build_router();
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("bind");
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("http://{}/", addr);
    let body = reqwest::get(&url).await.unwrap().text().await.unwrap();
    assert!(body.contains("LCA Webservice is running"));

    server.abort();
}

#[tokio::test]
async fn swagger_ui_serves() {
    let app = lca_webservice::app::build_router();
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("bind");
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let base = format!("http://{}", addr);
    let ui = reqwest::get(format!("{}/swagger-ui/", base)).await.unwrap();
    assert!(ui.status().is_success());
    let doc = reqwest::get(format!("{}/api-docs/openapi.json", base))
        .await
        .unwrap();
    assert!(doc.status().is_success());

    server.abort();
}
