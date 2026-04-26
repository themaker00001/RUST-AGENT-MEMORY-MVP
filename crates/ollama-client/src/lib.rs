use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: Client,
    base_url: String,
    embed_model: String,
    generate_model: String,
}

impl OllamaClient {
    pub fn new(
        base_url: impl Into<String>,
        embed_model: impl Into<String>,
        generate_model: impl Into<String>,
    ) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.into(),
            embed_model: embed_model.into(),
            generate_model: generate_model.into(),
        }
    }

    pub fn embed_model(&self) -> &str {
        &self.embed_model
    }

    fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    pub async fn embed(&self, input: &str) -> Result<Vec<f32>> {
        match self.embed_current_api(input).await {
            Ok(embedding) => Ok(embedding),
            Err(current_api_error) => self
                .embed_legacy_api(input)
                .await
                .with_context(|| {
                    format!(
                        "Ollama current embedding API failed first: {current_api_error}"
                    )
                }),
        }
    }

    async fn embed_current_api(&self, input: &str) -> Result<Vec<f32>> {
        let response = self
            .http
            .post(self.endpoint("/api/embed"))
            .json(&CurrentEmbeddingRequest {
                model: &self.embed_model,
                input,
            })
            .send()
            .await
            .context("failed to call Ollama /api/embed endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Ollama /api/embed request failed with {status}: {body}");
        }

        let body: CurrentEmbeddingResponse = response
            .json()
            .await
            .context("failed to parse Ollama /api/embed response")?;

        let embedding = body
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Ollama returned no embeddings"))?;

        if embedding.is_empty() {
            return Err(anyhow!("Ollama returned an empty embedding"));
        }

        Ok(embedding)
    }

    async fn embed_legacy_api(&self, input: &str) -> Result<Vec<f32>> {
        let response = self
            .http
            .post(self.endpoint("/api/embeddings"))
            .json(&LegacyEmbeddingRequest {
                model: &self.embed_model,
                prompt: input,
            })
            .send()
            .await
            .context("failed to call Ollama /api/embeddings endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Ollama /api/embeddings request failed with {status}: {body}");
        }

        let body: LegacyEmbeddingResponse = response
            .json()
            .await
            .context("failed to parse Ollama /api/embeddings response")?;

        if body.embedding.is_empty() {
            return Err(anyhow!("Ollama returned an empty embedding"));
        }

        Ok(body.embedding)
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let response = self
            .http
            .post(self.endpoint("/api/generate"))
            .json(&GenerateRequest {
                model: &self.generate_model,
                prompt,
                stream: false,
            })
            .send()
            .await
            .context("failed to call Ollama generate endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Ollama generate request failed with {status}: {body}");
        }

        let body: GenerateResponse = response
            .json()
            .await
            .context("failed to parse Ollama generate response")?;

        Ok(body.response.trim().to_string())
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
    ) -> Result<impl tokio_stream::Stream<Item = Result<String>>> {
        let response = self
            .http
            .post(self.endpoint("/api/generate"))
            .json(&GenerateRequest {
                model: &self.generate_model,
                prompt,
                stream: true,
            })
            .send()
            .await
            .context("failed to call Ollama generate endpoint")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Ollama generate request failed with {status}: {body}");
        }

        let stream = response.bytes_stream().map(|item| {
            let bytes = item.context("failed to read stream chunk")?;
            let s = String::from_utf8_lossy(&bytes);
            let mut combined_response = String::new();
            
            for line in s.lines() {
                if let Ok(body) = serde_json::from_str::<GenerateResponse>(line) {
                    combined_response.push_str(&body.response);
                }
            }
            Ok(combined_response)
        });

        Ok(stream)
    }
}


#[derive(Debug, Serialize)]
struct CurrentEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Debug, Deserialize)]
struct CurrentEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct LegacyEmbeddingRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Debug, Deserialize)]
struct LegacyEmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
}
