//! Background batch decoding with final-only output and whole-recording recovery.
//!
//! Only one inference runs at a time. Retain inexpensive PCM (1.92 MB/minute)
//! for the existing recovery workflow; model requests never exceed the cap.
use std::sync::Arc;

use anyhow::Result;
use audio_silence_gate::{audio_gate_reason, rms_energy, SILENCE_RMS_THRESHOLD};
use tokio::sync::mpsc;
use tracing::info;

use super::{TranscriptionBackend, TranscriptionConfig};
use crate::audio::capture::{encode_wav, SAMPLE_RATE};

pub type BatchResult = (Vec<i16>, Result<String>);
pub type BatchTask = tokio::task::JoinHandle<BatchResult>;

/// Choose the quietest 100ms in the last five seconds, cutting in its middle.
/// Samples on either side remain contiguous: no overlap or deduplication.
fn split_at(samples: &[i16], limit: usize) -> usize {
    let frame = SAMPLE_RATE as usize / 10;
    let start = limit.saturating_sub(5 * SAMPLE_RATE as usize).max(frame);
    let quiet = samples[start..limit]
        .chunks_exact(frame)
        .enumerate()
        .min_by(|(_, a), (_, b)| rms_energy(a).total_cmp(&rms_energy(b)))
        .map(|(index, _)| start + index * frame + frame / 2);
    quiet.unwrap_or(limit)
}

async fn decode(
    backend: &dyn TranscriptionBackend,
    config: &TranscriptionConfig,
    samples: &[i16],
) -> Result<String> {
    if audio_gate_reason(samples, SAMPLE_RATE, 300, SILENCE_RMS_THRESHOLD).is_some() {
        return Ok(String::new());
    }
    let started = std::time::Instant::now();
    let wav = encode_wav(samples)?;
    let text = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        backend.transcribe(&wav, config),
    )
    .await
    .map_err(|_| anyhow::anyhow!("background transcription timed out after 120 seconds"))??;
    info!(
        audio_seconds = samples.len() as f64 / SAMPLE_RATE as f64,
        elapsed_seconds = started.elapsed().as_secs_f64(),
        "background chunk transcribed"
    );
    if config
        .prompt
        .as_deref()
        .is_some_and(|p| prompt_echo::is_prompt_echo(&text, p))
    {
        return Ok(String::new());
    }
    Ok(text.trim().to_string())
}

/// Keep draining capture after a failed request so the complete recording can
/// be recovered. Never insert a silently incomplete transcript.
pub async fn run(
    mut rx: mpsc::UnboundedReceiver<Vec<i16>>,
    backend: Arc<dyn TranscriptionBackend>,
    config: TranscriptionConfig,
    seconds: u64,
) -> BatchResult {
    let limit = seconds.clamp(10, 120) as usize * SAMPLE_RATE as usize;
    let mut samples = Vec::new();
    let mut offset = 0;
    let mut parts = Vec::new();
    let mut failure = None;
    while let Some(chunk) = rx.recv().await {
        samples.extend(chunk);
        while failure.is_none() && samples.len() - offset >= limit {
            let end = offset + split_at(&samples[offset..], limit);
            match decode(backend.as_ref(), &config, &samples[offset..end]).await {
                Ok(text) if !text.is_empty() => parts.push(text),
                Ok(_) => {}
                Err(error) => failure = Some(error),
            }
            offset = end;
        }
    }
    if failure.is_none() && offset < samples.len() {
        match decode(backend.as_ref(), &config, &samples[offset..]).await {
            Ok(text) if !text.is_empty() => parts.push(text),
            Ok(_) => {}
            Err(error) => failure = Some(error),
        }
    }
    let result = failure.map_or_else(|| Ok(parts.join(" ")), Err);
    (samples, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Mock {
        calls: Mutex<Vec<Vec<i16>>>,
        fail: bool,
    }
    #[async_trait::async_trait]
    impl TranscriptionBackend for Mock {
        async fn transcribe(&self, audio: &[u8], _: &TranscriptionConfig) -> Result<String> {
            let reader = hound::WavReader::new(std::io::Cursor::new(audio))?;
            let samples = reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut calls = self.calls.lock().unwrap();
            calls.push(samples);
            if self.fail {
                anyhow::bail!("test failure");
            }
            Ok(format!("Part {}.", calls.len()))
        }
    }
    fn config() -> TranscriptionConfig {
        TranscriptionConfig {
            language: "en".into(),
            model: "test".into(),
            prompt: None,
        }
    }
    fn mock(fail: bool) -> Arc<Mock> {
        Arc::new(Mock {
            calls: Mutex::new(Vec::new()),
            fail,
        })
    }

    #[tokio::test]
    async fn decodes_before_stop_and_preserves_every_sample() {
        let backend = mock(false);
        let (tx, rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(rx, backend.clone(), config(), 30));
        let samples: Vec<i16> = (0..65 * SAMPLE_RATE as usize)
            .map(|i| 1000 + (i % 3000) as i16)
            .collect();
        tx.send(samples[..30 * SAMPLE_RATE as usize].to_vec())
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while backend.calls.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(!task.is_finished(), "waits for stop before returning text");
        tx.send(samples[30 * SAMPLE_RATE as usize..].to_vec())
            .unwrap();
        drop(tx);
        let (saved, text) = task.await.unwrap();
        assert_eq!(saved, samples);
        assert_eq!(text.unwrap(), "Part 1. Part 2. Part 3.");
        let calls = backend.calls.lock().unwrap();
        assert!(calls.iter().all(|c| c.len() <= 30 * SAMPLE_RATE as usize));
        assert_eq!(
            calls.concat(),
            samples,
            "no lost or duplicated boundary samples"
        );
    }

    #[tokio::test]
    async fn failure_retains_all_audio_and_returns_no_partial_success() {
        let backend = mock(true);
        let (tx, rx) = mpsc::unbounded_channel();
        let samples = vec![4000; 65 * SAMPLE_RATE as usize];
        for part in samples.chunks(1600) {
            tx.send(part.to_vec()).unwrap();
        }
        drop(tx);
        let (saved, result) = run(rx, backend.clone(), config(), 30).await;
        assert_eq!(saved, samples);
        assert!(result.unwrap_err().to_string().contains("test failure"));
        assert_eq!(backend.calls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn short_recording_is_one_request_and_silence_is_skipped() {
        for samples in [vec![3000; 16000], vec![0; 65 * 16000], vec![]] {
            let backend = mock(false);
            let (tx, rx) = mpsc::unbounded_channel();
            tx.send(samples.clone()).unwrap();
            drop(tx);
            let (saved, text) = run(rx, backend.clone(), config(), 30).await;
            assert_eq!(saved, samples);
            assert_eq!(
                backend.calls.lock().unwrap().len(),
                usize::from(samples.first() == Some(&3000))
            );
            assert!(text.is_ok());
        }
    }

    #[test]
    fn prefers_pause_near_boundary() {
        let mut samples = vec![4000; 30 * 16000];
        samples[28 * 16000..28 * 16000 + 1600].fill(0);
        assert_eq!(split_at(&samples, samples.len()), 28 * 16000 + 800);
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_preserves_audio_for_recovery() {
        struct Hanging;
        #[async_trait::async_trait]
        impl TranscriptionBackend for Hanging {
            async fn transcribe(&self, _: &[u8], _: &TranscriptionConfig) -> Result<String> {
                std::future::pending().await
            }
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let samples = vec![4000; 16000];
        tx.send(samples.clone()).unwrap();
        drop(tx);
        let (saved, result) = run(rx, Arc::new(Hanging), config(), 30).await;
        assert_eq!(saved, samples);
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn abort_during_request_closes_capture_receiver() {
        struct Hanging(tokio::sync::Notify);
        #[async_trait::async_trait]
        impl TranscriptionBackend for Hanging {
            async fn transcribe(&self, _: &[u8], _: &TranscriptionConfig) -> Result<String> {
                self.0.notify_one();
                std::future::pending().await
            }
        }
        let backend = Arc::new(Hanging(tokio::sync::Notify::new()));
        let (tx, rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(rx, backend.clone(), config(), 30));
        tx.send(vec![4000; 30 * 16000]).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), backend.0.notified())
            .await
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(tx.send(vec![4000; 1600]).is_err());
    }
}
