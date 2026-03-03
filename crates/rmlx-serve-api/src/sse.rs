//! SSE (Server-Sent Events) streaming utilities.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_core::Stream;
use futures_util::StreamExt;

use rmlx_serve_types::openai::ChatCompletionChunk;

/// Wrap a `Stream<Item = String>` into an Axum `Sse` response.
///
/// Each item from the stream is sent as an SSE `data:` event.
pub fn create_sse_stream<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = String> + Send + 'static,
{
    Sse::new(stream.map(|data| Ok(Event::default().data(data))))
}

/// Serialise a [`ChatCompletionChunk`] to the SSE data line format.
///
/// Returns the JSON string (without the `data: ` prefix or trailing newlines,
/// as Axum's SSE layer adds those).
pub fn format_chat_chunk(chunk: &ChatCompletionChunk) -> String {
    serde_json::to_string(chunk).unwrap_or_else(|_| "{}".to_string())
}

/// The sentinel value that signals end-of-stream in the OpenAI SSE protocol.
pub const SSE_DONE: &str = "[DONE]";
