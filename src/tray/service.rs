//! System tray implementation using ksni (StatusNotifierItem).

use std::io::Write;
use std::sync::{Arc, RwLock};

use anyhow::Context;
use ksni::{Icon, ToolTip, TrayMethods};
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::audio::capture::InputDeviceChoice;
use crate::{DictationPreferences, OutputMode, State};

/// 16x16 ARGB icon data for each state.
/// Format: each pixel is 4 bytes (ARGB, big-endian).
mod icons {
    /// Generate a simple 16x16 solid circle icon with the given ARGB color.
    pub fn circle_icon(argb: u32) -> Vec<u8> {
        let size = 16;
        let center = size as f32 / 2.0;
        let radius = 6.0;
        let mut pixels = Vec::with_capacity(size * size * 4);

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 + 0.5 - center;
                let dy = y as f32 + 0.5 - center;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist <= radius {
                    pixels.extend_from_slice(&argb.to_be_bytes());
                } else if dist <= radius + 1.0 {
                    let alpha = ((radius + 1.0 - dist) * 255.0) as u8;
                    let [_, r, g, b] = argb.to_be_bytes();
                    pixels.extend_from_slice(&[alpha, r, g, b]);
                } else {
                    pixels.extend_from_slice(&[0, 0, 0, 0]);
                }
            }
        }
        pixels
    }

    pub fn idle() -> Vec<u8> {
        circle_icon(0xFF_88_88_88)
    }

    pub fn recording() -> Vec<u8> {
        circle_icon(0xFF_E0_40_40)
    }

    pub fn transcribing() -> Vec<u8> {
        circle_icon(0xFF_E0_A0_20)
    }

    /// Read-aloud: synthesizing speech (blue/purple).
    pub fn synthesizing() -> Vec<u8> {
        circle_icon(0xFF_7C_5C_FF)
    }

    /// Read-aloud: playing speech (green).
    pub fn speaking() -> Vec<u8> {
        circle_icon(0xFF_34_D3_99)
    }
}

/// Small mutable state owned by the tray service itself.
///
/// Keeping this directly on the tray object is important: `ksni::Handle::update`
/// expects the closure to mutate the tray instance so the host knows which
/// properties changed. When the state lives out-of-band, some tray hosts can
/// miss icon refreshes and leave the old color visible.
struct TrayState {
    current: State,
}

/// The ksni tray implementation.
struct WhisrsTray {
    state: TrayState,
    preferences: Arc<RwLock<DictationPreferences>>,
    audio_devices: Vec<InputDeviceChoice>,
}

impl ksni::Tray for WhisrsTray {
    // A click on the top-bar icon should expose the controls directly.
    const MENU_ON_ACTIVATE: bool = true;

    fn id(&self) -> String {
        "whisrs".to_string()
    }

    fn title(&self) -> String {
        match self.state.current {
            State::Idle => "whisrs — idle".to_string(),
            State::Recording => "whisrs — recording".to_string(),
            State::Transcribing => "whisrs — transcribing".to_string(),
            State::Synthesizing => "whisrs — synthesizing".to_string(),
            State::Speaking => "whisrs — speaking".to_string(),
        }
    }

    fn icon_pixmap(&self) -> Vec<Icon> {
        let data = match self.state.current {
            State::Idle => icons::idle(),
            State::Recording => icons::recording(),
            State::Transcribing => icons::transcribing(),
            State::Synthesizing => icons::synthesizing(),
            State::Speaking => icons::speaking(),
        };
        vec![Icon {
            width: 16,
            height: 16,
            data,
        }]
    }

    fn tool_tip(&self) -> ToolTip {
        let state_description = match self.state.current {
            State::Idle => "Idle — ready to record",
            State::Recording => "Recording...",
            State::Transcribing => "Transcribing...",
            State::Synthesizing => "Synthesizing…",
            State::Speaking => "Reading aloud…",
        };
        ToolTip {
            title: "whisrs".to_string(),
            description: match self.preferences.read() {
                Ok(preferences) => format!(
                    "{state_description}\nOutput: {} · Microphone: {}",
                    preferences.output_mode,
                    self.device_label(&preferences.audio_device)
                ),
                Err(_) => state_description.to_string(),
            },
            icon_name: String::new(),
            icon_pixmap: Vec::new(),
        }
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{RadioGroup, RadioItem, StandardItem};

        let preferences = self
            .preferences
            .read()
            .map(|settings| settings.clone())
            .unwrap_or_else(|_| DictationPreferences {
                output_mode: OutputMode::Type,
                audio_device: "default".to_string(),
            });
        let enabled = self.state.current == State::Idle;
        let selected_device = self
            .audio_devices
            .iter()
            .position(|device| device.id == preferences.audio_device)
            .unwrap_or(0);
        let has_transcription = crate::history::read_entries(1)
            .map(|entries| !entries.is_empty())
            .unwrap_or(false);

        vec![
            StandardItem {
                label: format!(
                    "Microphone — {}",
                    self.device_label(&preferences.audio_device)
                ),
                enabled: false,
                ..Default::default()
            }
            .into(),
            RadioGroup {
                selected: selected_device,
                select: Box::new(|tray: &mut Self, selected| {
                    let Some(device) = tray
                        .audio_devices
                        .get(selected)
                        .map(|choice| choice.id.clone())
                    else {
                        return;
                    };
                    tray.update_preferences(|preferences| {
                        preferences.audio_device = device;
                    });
                }),
                options: self
                    .audio_devices
                    .iter()
                    .map(|device| RadioItem {
                        label: device.label.clone(),
                        enabled,
                        ..Default::default()
                    })
                    .collect(),
            }
            .into(),
            ksni::MenuItem::Separator,
            StandardItem {
                label: format!("Output — {}", preferences.output_mode),
                enabled: false,
                ..Default::default()
            }
            .into(),
            RadioGroup {
                selected: match preferences.output_mode {
                    OutputMode::Type => 0,
                    OutputMode::Paste => 1,
                },
                select: Box::new(|tray: &mut Self, selected| {
                    let mode = if selected == 1 {
                        OutputMode::Paste
                    } else {
                        OutputMode::Type
                    };
                    tray.update_preferences(|preferences| {
                        preferences.output_mode = mode;
                    });
                }),
                options: vec![
                    RadioItem {
                        label: "Type individual keys".to_string(),
                        enabled,
                        ..Default::default()
                    },
                    RadioItem {
                        label: "Paste all at once".to_string(),
                        enabled,
                        ..Default::default()
                    },
                ],
            }
            .into(),
            ksni::MenuItem::Separator,
            StandardItem {
                label: "Cancel current transcription".to_string(),
                enabled: matches!(self.state.current, State::Recording | State::Transcribing),
                activate: Box::new(|_| {
                    tokio::spawn(async {
                        use tokio::io::AsyncWriteExt;
                        let result = async {
                            let mut stream =
                                tokio::net::UnixStream::connect(crate::socket_path()).await?;
                            stream
                                .write_all(&crate::encode_message(&crate::Command::Cancel)?)
                                .await?;
                            stream.shutdown().await?;
                            let response: crate::Response =
                                crate::read_message(&mut stream).await?;
                            if let crate::Response::Error { message } = response {
                                anyhow::bail!(message);
                            }
                            Ok::<_, anyhow::Error>(())
                        }
                        .await;
                        if let Err(error) = result {
                            warn!("could not cancel transcription: {error:#}");
                        }
                    });
                }),
                ..Default::default()
            }
            .into(),
            ksni::MenuItem::Separator,
            StandardItem {
                label: "Copy last transcription".to_string(),
                enabled: has_transcription,
                icon_name: "edit-copy".to_string(),
                activate: Box::new(|_| copy_last_transcription()),
                ..Default::default()
            }
            .into(),
        ]
    }
}

/// Copy the most recent persisted transcription without blocking the tray's
/// D-Bus menu callback while the clipboard backend does its work.
fn copy_last_transcription() {
    std::thread::spawn(|| {
        let result = (|| -> anyhow::Result<String> {
            let entry = crate::history::read_entries(1)?
                .into_iter()
                .next()
                .context("no previous transcription is available yet")?;
            xkb_type::default_clipboard()
                .set_text(&entry.text)
                .context("failed to copy the last transcription")?;
            Ok(entry.text)
        })();

        match result {
            Ok(text) => {
                info!(
                    "copied last transcription to clipboard ({} chars)",
                    text.len()
                );
                show_copy_notification("Last transcription copied", "Ready to paste.");
            }
            Err(error) => {
                warn!("could not copy last transcription: {error:#}");
                show_copy_notification("Could not copy last transcription", &error.to_string());
            }
        }
    });
}

fn show_copy_notification(summary: &str, body: &str) {
    if let Err(error) = notify_rust::Notification::new()
        .summary(summary)
        .body(body)
        .appname("whisrs")
        .timeout(notify_rust::Timeout::Milliseconds(2000))
        .show()
    {
        warn!("failed to show tray notification: {error}");
    }
}

impl WhisrsTray {
    fn device_label(&self, id: &str) -> String {
        self.audio_devices
            .iter()
            .find(|choice| choice.id == id)
            .map(|choice| choice.label.clone())
            .unwrap_or_else(|| id.to_string())
    }

    fn update_preferences(&self, update: impl FnOnce(&mut DictationPreferences)) {
        let snapshot = match self.preferences.write() {
            Ok(mut preferences) => {
                update(&mut preferences);
                preferences.clone()
            }
            Err(_) => {
                warn!("could not update tray preferences: lock poisoned");
                return;
            }
        };

        if let Err(error) = persist_preferences(&snapshot) {
            warn!("could not persist tray preferences: {error:#}");
        } else {
            info!(
                "dictation preferences updated (output={}, microphone={})",
                snapshot.output_mode, snapshot.audio_device
            );
        }
    }
}

/// Update only the tray-controlled keys while preserving comments, API keys,
/// and the rest of the user's TOML formatting.
fn persist_preferences(preferences: &DictationPreferences) -> anyhow::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;

    let config_path = crate::config_path();
    let source = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let updated = update_preferences_document(&source, preferences)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    let parent = config_path
        .parent()
        .context("config path does not have a parent directory")?;
    let temp_name = format!(
        ".config.toml.whisrs-{}-{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let temp_path = parent.join(temp_name);
    let write_result = (|| -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp_path)
            .with_context(|| format!("failed to create {}", temp_path.display()))?;
        file.write_all(updated.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&temp_path, &config_path)
            .with_context(|| format!("failed to replace {}", config_path.display()))?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    write_result
}

fn update_preferences_document(
    source: &str,
    preferences: &DictationPreferences,
) -> anyhow::Result<String> {
    let mut document = source.parse::<toml_edit::DocumentMut>()?;
    set_string_preserving_decor(
        &mut document["input"]["output_mode"],
        &preferences.output_mode.to_string(),
    );
    set_string_preserving_decor(&mut document["audio"]["device"], &preferences.audio_device);
    Ok(document.to_string())
}

fn set_string_preserving_decor(item: &mut toml_edit::Item, new_value: &str) {
    let decor = item.as_value().map(|value| value.decor().clone());
    *item = toml_edit::value(new_value);
    if let (Some(decor), Some(value)) = (decor, item.as_value_mut()) {
        *value.decor_mut() = decor;
    }
}

/// Maximum number of attempts to connect to the SNI tray host.
const TRAY_MAX_RETRIES: u32 = 10;

/// Initial retry delay (doubles each attempt, capped at 10 s).
const TRAY_INITIAL_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

/// Spawn the system tray indicator.
///
/// Runs in the background and updates the icon whenever the daemon state changes.
/// Retries with exponential backoff if the SNI host isn't available yet (common
/// on boot when the daemon starts before the desktop environment is fully ready).
pub async fn spawn_tray(
    mut state_rx: watch::Receiver<State>,
    preferences: Arc<RwLock<DictationPreferences>>,
    audio_devices: Vec<InputDeviceChoice>,
) {
    // Retry spawning the tray with exponential backoff.
    let mut delay = TRAY_INITIAL_DELAY;
    let mut handle = None;

    for attempt in 1..=TRAY_MAX_RETRIES {
        let tray = WhisrsTray {
            state: TrayState {
                current: *state_rx.borrow(),
            },
            preferences: Arc::clone(&preferences),
            audio_devices: audio_devices.clone(),
        };

        match tray.spawn().await {
            Ok(h) => {
                info!("system tray started (attempt {attempt})");
                handle = Some(h);
                break;
            }
            Err(e) => {
                if attempt == TRAY_MAX_RETRIES {
                    warn!(
                        "failed to start system tray after {TRAY_MAX_RETRIES} attempts: {e} — continuing without tray"
                    );
                    return;
                }
                info!(
                    "tray host not available (attempt {attempt}/{TRAY_MAX_RETRIES}): {e} — retrying in {delay:?}"
                );
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(10));
            }
        }
    }

    let handle = handle.expect("handle must be set after successful spawn");

    // Watch for state changes and update the tray.
    tokio::spawn(async move {
        while state_rx.changed().await.is_ok() {
            let new_state = *state_rx.borrow();
            debug!("tray state update: {new_state:?}");
            // Mutate the tray object itself so ksni emits the corresponding
            // D-Bus property changes for title, tooltip, and icon pixmap.
            handle
                .update(|tray| {
                    tray.state.current = new_state;
                })
                .await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_separate_and_only_enabled_during_dictation() {
        use ksni::Tray;
        for state in [
            State::Idle,
            State::Recording,
            State::Transcribing,
            State::Speaking,
            State::Synthesizing,
        ] {
            let tray = WhisrsTray {
                state: TrayState { current: state },
                preferences: Arc::new(RwLock::new(DictationPreferences {
                    output_mode: OutputMode::Paste,
                    audio_device: "default".into(),
                })),
                audio_devices: vec![],
            };
            let menu = tray.menu();
            let index = menu.iter().position(|item| matches!(item, ksni::MenuItem::Standard(item) if item.label == "Cancel current transcription")).unwrap();
            let ksni::MenuItem::Standard(item) = &menu[index] else {
                unreachable!()
            };
            assert_eq!(
                item.enabled,
                matches!(state, State::Recording | State::Transcribing)
            );
            assert!(matches!(menu[index - 1], ksni::MenuItem::Separator));
            assert!(matches!(menu[index + 1], ksni::MenuItem::Separator));
        }
    }

    #[test]
    fn preference_update_preserves_unrelated_config_and_comments() {
        let source = r#"# keep this comment
[general]
backend = "groq"

[groq]
api_key = "secret-value"

[audio]
device = "default" # existing comment

[input]
key_delay_ms = 7
"#;
        let updated = update_preferences_document(
            source,
            &DictationPreferences {
                output_mode: OutputMode::Paste,
                audio_device: "USB Microphone".to_string(),
            },
        )
        .unwrap();

        assert!(updated.contains("# keep this comment"));
        assert!(updated.contains("api_key = \"secret-value\""));
        assert!(updated.contains("device = \"USB Microphone\" # existing comment"));
        assert!(updated.contains("output_mode = \"paste\""));
        assert!(updated.contains("key_delay_ms = 7"));
    }
}
