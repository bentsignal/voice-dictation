//! Desktop overlay shown while recording or transcribing.

use std::os::fd::AsRawFd;
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
enum OverlayError {
    #[error("Wayland connection error: {0}")]
    Connect(#[from] wayland_client::ConnectError),
    #[error("Wayland I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Wayland backend error: {0}")]
    Backend(#[from] wayland_client::backend::WaylandError),
    #[error("Wayland globals error: {0}")]
    Globals(#[from] wayland_client::globals::GlobalError),
    #[error("smithay bind error: {0}")]
    Bind(#[from] wayland_client::globals::BindError),
    #[error("smithay shm create error: {0}")]
    Shm(#[from] smithay_client_toolkit::shm::CreatePoolError),
    #[error("Wayland dispatch error: {0}")]
    Dispatch(#[from] wayland_client::DispatchError),
    #[error("X11 display connect error: {0}")]
    X11Connect(#[from] x11rb::errors::ConnectError),
    #[error("X11 protocol/IO error: {0}")]
    X11Connection(#[from] x11rb::errors::ConnectionError),
    #[error("X11 reply error: {0}")]
    X11Reply(#[from] x11rb::errors::ReplyError),
    #[error("X11 reply/id error: {0}")]
    X11ReplyOrId(#[from] x11rb::errors::ReplyOrIdError),
    #[error("X11 ARGB visual not available")]
    X11Visual,
    #[error("D-Bus error: {0}")]
    DBus(#[from] zbus::Error),
    #[error("D-Bus signal error: {0}")]
    DBusSignal(#[from] zbus::fdo::Error),
    #[error("tiny-skia pixmap allocation failed for {0}x{1}")]
    Pixmap(u32, u32),
}

use smithay_client_toolkit::reexports::protocols::wp::viewporter::client::{
    wp_viewport::{self, WpViewport},
    wp_viewporter::{self, WpViewporter},
};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{slot::SlotPool, Shm, ShmHandler},
};
use tiny_skia::{
    Color, FillRule, Paint, PathBuilder, Pixmap, PremultipliedColorU8, Rect, Transform,
};
use tokio::sync::watch;
use tracing::{info, warn};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_region, wl_shm, wl_surface},
    Connection as WaylandConnection, Dispatch, QueueHandle,
};
use x11rb::connection::{Connection as X11Connection, RequestConnection};
use x11rb::properties::WmHints;
use x11rb::protocol::{
    randr::{ConnectionExt as X11RandrConnectionExt, MonitorInfo, X11_EXTENSION_NAME as RANDR_EXT},
    shape::{ConnectionExt as X11ShapeConnectionExt, SK, SO, X11_EXTENSION_NAME as SHAPE_EXT},
    xproto::{
        AtomEnum, ClipOrdering, ColormapAlloc, ConfigureWindowAux,
        ConnectionExt as X11ProtoConnectionExt, CreateGCAux, CreateWindowAux, EventMask,
        ImageFormat, PropMode, Rectangle, Screen, StackMode, VisualClass, Visualid, Window,
        WindowClass,
    },
};
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as X11WrapperConnectionExt;

use crate::{OverlayConfig, State};

const BOTTOM_MARGIN: i32 = 16;

// Per-frame sleep matching the draw loop. ~16 ms ≈ 60 fps for visibly
// smoother motion. Spawn animation progress is wall-clock-driven (see
// `Overlay::spawn_t`), so this only caps the redraw rate.
const FRAME_MS: u64 = 16;

// Spawn animation: the pill "draws out" from a 4-px sliver anchored at the
// bottom of the surface up to its full configured height. Slight overshoot
// for life. Going away is shorter and accelerated — feels intentional.
const SPAWN_IN_MS: f32 = 220.0;
const SPAWN_OUT_MS: f32 = 140.0;
/// Initial pill height during the appear animation, in px.
const SPAWN_PILL_MIN_H: f32 = 4.0;
/// `easeOutBack` overshoot constant. 0.4 ⇒ peak ~3 % over target before
/// settling — barely perceptible but adds a "physical arrival" feel.
const SPAWN_OVERSHOOT_C: f32 = 0.4;

/// While the pill is still growing, bars stay fully invisible for this many
/// ms after appearance — then they fade in over `BARS_FADE_MS` while the
/// pill finishes settling. After both have elapsed, audio reactivity
/// unlocks.
const BARS_GRACE_MS: f32 = 80.0;
const BARS_FADE_MS: f32 = 80.0;

// Bar layout. 7 bars × 3 px + 6 gaps × 2 px = 33 px wide, centered in
// the pill. More, thinner bars means motion reads as a continuous
// equalizer ripple instead of a few chunky blocks. Max bar height =
// HEIGHT − 2·BAR_VPAD (e.g. 28 px on the default 40 px pill). Bar height
// is purely level-driven — no per-bar phase animation — so each bar
// stays anchored at the pill center and expands symmetrically up and
// down with the audio amplitude.
const BAR_COUNT: u32 = 7;
const BAR_W: f32 = 3.0;
const BAR_GAP: f32 = 2.0;
const BAR_BASELINE: f32 = 6.0;
const BAR_VPAD: f32 = 6.0;

/// Color palette for one overlay theme. Bytes are stored as `[A, R, G, B]`,
/// matching the canvas pixel layout used by [`blend_pixel`].
#[derive(Debug, Clone, Copy)]
struct Theme {
    bg: [u8; 4],
    ring: [u8; 4],
    rec_bar: [u8; 4],
    trans_bar: [u8; 4],
    /// Read-aloud "speaking" bar color — a distinct hue from `rec_bar` so the
    /// user can tell dictation from read-aloud at a glance.
    speak_bar: [u8; 4],
    glow: [u8; 4],
}

impl Theme {
    /// Default palette — warm "tally light" amber on near-black slate.
    const fn ember() -> Self {
        Self {
            bg: [235, 14, 14, 16],           // #0E0E10 @ 92%
            ring: [64, 249, 115, 22],        // #F97316 @ 25%
            rec_bar: [255, 249, 115, 22],    // #F97316
            trans_bar: [255, 240, 237, 245], // #F0EDF5
            speak_bar: [255, 52, 211, 153],  // #34D399 emerald — distinct from amber
            glow: [60, 249, 115, 22],
        }
    }

    /// Monochrome terminal palette — subdued, never distracting.
    const fn carbon() -> Self {
        Self {
            bg: [235, 14, 14, 16],
            ring: [80, 58, 58, 64],          // hairline gray
            rec_bar: [255, 240, 237, 245],   // soft white
            trans_bar: [255, 156, 163, 175], // warm gray
            speak_bar: [255, 45, 212, 191],  // #2DD4BF teal — pops against the grays
            glow: [40, 240, 237, 245],
        }
    }

    /// Cool electric-blue palette — audio-equipment vibe.
    const fn cyan() -> Self {
        Self {
            bg: [235, 10, 15, 20],
            ring: [64, 34, 211, 238], // #22D3EE @ 25%
            rec_bar: [255, 34, 211, 238],
            trans_bar: [255, 56, 189, 248], // #38BDF8
            speak_bar: [255, 74, 222, 128], // #4ADE80 green — distinct from cyan
            glow: [50, 34, 211, 238],
        }
    }

    fn from_config(cfg: &OverlayConfig) -> Self {
        let base = match cfg.theme.as_str() {
            "carbon" => Self::carbon(),
            "cyan" => Self::cyan(),
            "ember" | "custom" => Self::ember(),
            other => {
                warn!("unknown overlay theme {other:?}, falling back to ember");
                Self::ember()
            }
        };
        if cfg.theme != "custom" {
            return base;
        }
        let Some(c) = cfg.colors.as_ref() else {
            return base;
        };
        Self {
            bg: c
                .background
                .as_deref()
                .and_then(crate::parse_hex_color)
                .unwrap_or(base.bg),
            ring: c
                .ring
                .as_deref()
                .and_then(crate::parse_hex_color)
                .unwrap_or(base.ring),
            rec_bar: c
                .recording
                .as_deref()
                .and_then(crate::parse_hex_color)
                .unwrap_or(base.rec_bar),
            trans_bar: c
                .transcribing
                .as_deref()
                .and_then(crate::parse_hex_color)
                .unwrap_or(base.trans_bar),
            speak_bar: c
                .speaking
                .as_deref()
                .and_then(crate::parse_hex_color)
                .unwrap_or(base.speak_bar),
            glow: c
                .glow
                .as_deref()
                .and_then(crate::parse_hex_color)
                .unwrap_or(base.glow),
        }
    }
}

/// Spawn the bottom recording overlay.
///
/// Native compositor event loops run on a dedicated OS thread because they are
/// blocking client loops. A small Tokio task forwards daemon state changes into
/// that thread.
pub async fn spawn_overlay(
    state_rx: watch::Receiver<State>,
    level_rx: watch::Receiver<f32>,
    config: OverlayConfig,
) {
    if is_gnome_desktop() {
        let gnome_state_rx = state_rx.clone();
        let gnome_level_rx = level_rx.clone();
        let gnome_theme = config.theme.clone();
        tokio::spawn(async move {
            if let Err(e) = run_gnome_broadcaster(gnome_state_rx, gnome_level_rx, gnome_theme).await
            {
                warn!("GNOME overlay D-Bus broadcaster unavailable: {e:#}");
            }
        });
    }

    loop {
        run_overlay_session(state_rx.clone(), level_rx.clone(), config.clone()).await;
        if state_rx.has_changed().is_err() {
            break;
        }
        warn!("overlay connection or layer closed; recreating overlay in one second");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn run_overlay_session(
    mut state_rx: watch::Receiver<State>,
    mut level_rx: watch::Receiver<f32>,
    config: OverlayConfig,
) {
    let (tx, rx) = mpsc::channel::<State>();
    let (level_tx, level_rx_thread) = mpsc::channel::<f32>();
    let (focused_output_tx, focused_output_rx) = mpsc::channel::<String>();
    let watcher = spawn_hyprland_focused_output_watcher(state_rx.clone(), focused_output_tx);
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

    let backend = OverlayBackend::detect();
    info!("overlay backend selected: {backend:?}");
    let overlay_config = config;
    std::thread::Builder::new()
        .name("whisrs-overlay".to_string())
        .spawn(move || {
            // Dropping the sender also wakes the supervisor on an early return.
            let _done = done_tx;
            let result = match backend {
                OverlayBackend::Wayland => {
                    run_overlay(rx, level_rx_thread, focused_output_rx, overlay_config)
                }
                OverlayBackend::X11 => run_x11_overlay(rx, level_rx_thread, overlay_config),
                OverlayBackend::Unavailable => {
                    warn!("overlay unavailable: no Wayland or X11 display in environment");
                    return;
                }
            };
            if let Err(e) = result {
                warn!("overlay unavailable: {e:#}");
            }
        })
        .map_err(|e| warn!("failed to spawn overlay thread: {e}"))
        .ok();

    let forwarding = tokio::spawn(async move {
        let _ = tx.send(*state_rx.borrow());
        let _ = level_tx.send(*level_rx.borrow());
        loop {
            tokio::select! {
                changed = state_rx.changed() => {
                    if changed.is_err() { break; }
                    if tx.send(*state_rx.borrow()).is_err() { break; }
                }
                changed = level_rx.changed() => {
                    if changed.is_err() { break; }
                    let _ = level_tx.send(*level_rx.borrow());
                }
            }
        }
    });
    let _ = done_rx.await;
    forwarding.abort();
    if let Some(watcher) = watcher {
        watcher.abort();
    }
}

fn spawn_hyprland_focused_output_watcher(
    mut state_rx: watch::Receiver<State>,
    focused_output_tx: mpsc::Sender<String>,
) -> Option<tokio::task::JoinHandle<()>> {
    if !env_var_is_set("HYPRLAND_INSTANCE_SIGNATURE") {
        return None;
    }

    Some(tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(200));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut active = *state_rx.borrow() != State::Idle;
        let mut last_output = None;

        loop {
            let should_query = tokio::select! {
                changed = state_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let was_active = active;
                    active = *state_rx.borrow() != State::Idle;
                    active && !was_active
                }
                _ = interval.tick(), if active => true,
            };
            if !should_query {
                continue;
            }

            let Some(output_name) = query_hyprland_focused_output().await else {
                continue;
            };
            if last_output.as_deref() == Some(output_name.as_str()) {
                continue;
            }
            last_output = Some(output_name.clone());
            if focused_output_tx.send(output_name).is_err() {
                break;
            }
        }
    }))
}

async fn query_hyprland_focused_output() -> Option<String> {
    let output = tokio::process::Command::new("hyprctl")
        .args(["activeworkspace", "-j"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_hyprland_active_workspace_output(&output.stdout)
}

fn parse_hyprland_active_workspace_output(stdout: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(stdout).ok()?;
    value
        .get("monitor")?
        .as_str()
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

#[derive(Debug, Clone, Copy)]
enum OverlayBackend {
    Wayland,
    X11,
    Unavailable,
}

impl OverlayBackend {
    fn detect() -> Self {
        let session_is_wayland = matches_env("XDG_SESSION_TYPE", "wayland");
        let session_is_x11 = matches_env("XDG_SESSION_TYPE", "x11");
        // Plasma/KWin does not expose the wlroots layer-shell protocol used by
        // the native Wayland renderer.  KWin does, however, host XWayland and
        // the X11 renderer is an override-redirect, click-through notification
        // window, which gives us the same bottom-centre HUD without stealing
        // focus.
        if is_kde_desktop() && env_var_is_set("DISPLAY") {
            Self::X11
        } else if (env_var_is_set("WAYLAND_DISPLAY") || session_is_wayland) && !session_is_x11 {
            Self::Wayland
        } else if env_var_is_set("DISPLAY") {
            Self::X11
        } else {
            Self::Unavailable
        }
    }
}

fn env_var_is_set(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| !value.is_empty())
}

fn matches_env(name: &str, expected: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value.eq_ignore_ascii_case(expected))
}

fn is_kde_desktop() -> bool {
    std::env::var("XDG_CURRENT_DESKTOP")
        .map(|value| {
            value
                .split(':')
                .any(|part| part.eq_ignore_ascii_case("KDE"))
        })
        .unwrap_or(false)
}

fn is_gnome_desktop() -> bool {
    std::env::var("XDG_CURRENT_DESKTOP")
        .map(|value| {
            value
                .split(':')
                .any(|part| part.eq_ignore_ascii_case("GNOME"))
        })
        .unwrap_or(false)
}

async fn run_gnome_broadcaster(
    mut state_rx: watch::Receiver<State>,
    level_rx: watch::Receiver<f32>,
    theme: String,
) -> Result<(), OverlayError> {
    // Custom themes don't sync over D-Bus for v1 — the GNOME extension only
    // knows the named themes it ships. Fall back to "ember" so the bar
    // colors remain sensible.
    let advertised_theme = match theme.as_str() {
        "carbon" | "cyan" | "ember" => theme.clone(),
        _ => "ember".to_string(),
    };

    let conn = zbus::connection::Builder::session()?
        .serve_at("/org/whisrs/Overlay", GnomeOverlayBus)?
        .name("org.whisrs.Overlay")?
        .build()
        .await?;

    info!("GNOME overlay D-Bus broadcaster started");
    emit_gnome_theme(&conn, &advertised_theme).await?;
    let initial_state = *state_rx.borrow();
    emit_gnome_state(&conn, initial_state).await?;
    let initial_level = *level_rx.borrow();
    emit_gnome_level(&conn, initial_level).await?;

    // Emit level at a steady ~30 Hz so the GNOME extension always has fresh
    // data, regardless of how the watch channel coalesces updates.
    let mut level_interval = tokio::time::interval(Duration::from_millis(33));
    level_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            changed = state_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let state = *state_rx.borrow();
                emit_gnome_state(&conn, state).await?;
            }
            _ = level_interval.tick() => {
                let level = *level_rx.borrow();
                emit_gnome_level(&conn, level).await?;
            }
        }
    }

    Ok(())
}

async fn emit_gnome_state(conn: &zbus::Connection, state: State) -> zbus::Result<()> {
    conn.emit_signal(
        None::<&str>,
        "/org/whisrs/Overlay",
        "org.whisrs.Overlay",
        "StateChanged",
        &(state.to_string()),
    )
    .await
}

async fn emit_gnome_level(conn: &zbus::Connection, level: f32) -> zbus::Result<()> {
    conn.emit_signal(
        None::<&str>,
        "/org/whisrs/Overlay",
        "org.whisrs.Overlay",
        "LevelChanged",
        &level.clamp(0.0, 1.0),
    )
    .await
}

async fn emit_gnome_theme(conn: &zbus::Connection, theme: &str) -> zbus::Result<()> {
    conn.emit_signal(
        None::<&str>,
        "/org/whisrs/Overlay",
        "org.whisrs.Overlay",
        "ThemeChanged",
        &theme,
    )
    .await
}

struct GnomeOverlayBus;

#[zbus::interface(name = "org.whisrs.Overlay")]
impl GnomeOverlayBus {
    fn ping(&self) -> &'static str {
        "ok"
    }
}

fn run_overlay(
    state_rx: mpsc::Receiver<State>,
    level_rx: mpsc::Receiver<f32>,
    focused_output_rx: mpsc::Receiver<String>,
    config: OverlayConfig,
) -> Result<(), OverlayError> {
    let width = config.clamped_width();
    let height = config.clamped_height();
    let sensitivity = config.clamped_sensitivity();
    let theme = Theme::from_config(&config);

    let conn = WaylandConnection::connect_to_env()?;
    let (globals, mut event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let compositor = CompositorState::bind(&globals, &qh)?;
    let layer_shell = LayerShell::bind(&globals, &qh)?;
    let shm = Shm::bind(&globals, &qh)?;

    // Viewporter lets us submit a 2x render buffer while retaining the exact
    // logical layer-shell size. It is optional so the overlay still works on
    // older compositors, just without HiDPI supersampling.
    let viewporter = globals.bind(&qh, 1..=1, ()).ok();
    let (layer, viewport) = create_overlay_layer(
        &compositor,
        &layer_shell,
        viewporter.as_ref(),
        &qh,
        width,
        height,
        None,
    );

    let pool = SlotPool::new((width * height * 4) as usize, &shm)?;
    let mut overlay = Overlay {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        compositor,
        layer_shell,
        viewporter,
        viewport,
        shm,
        pool,
        layer,
        focused_output_rx,
        desired_output_name: None,
        current_output_name: None,
        renderer: OverlayRenderer::new(state_rx, level_rx, width, height, theme, sensitivity)?,
        logical_width: width,
        logical_height: height,
        exit: false,
        first_configure: true,
    };

    info!("recording overlay started");
    let mut next_frame = Instant::now();
    let mut needs_clear = true;
    while !overlay.exit {
        overlay.renderer.apply_state_updates();
        overlay.apply_focused_output(&qh);
        if overlay.renderer.disconnected {
            break;
        }
        // Invisible/occluded surfaces may never receive frame callbacks.
        // Poll with a deadline so recording state and audio always wake the HUD.
        event_queue.dispatch_pending(&mut overlay)?;
        if overlay.exit {
            break;
        }
        let now = Instant::now();
        if now >= next_frame {
            if !overlay.first_configure
                && (overlay.renderer.target_state != State::Idle || needs_clear)
            {
                overlay.draw(&qh);
                needs_clear = overlay.renderer.target_state != State::Idle;
            }
            next_frame = now + Duration::from_millis(FRAME_MS);
        }
        tolerate_would_block(conn.flush())?;
        if let Some(guard) = event_queue.prepare_read() {
            let mut fd = libc::pollfd {
                fd: guard.connection_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: fd points to one initialized pollfd for this call.
            let timeout = next_frame
                .saturating_duration_since(Instant::now())
                .as_millis()
                .max(1) as i32;
            let ready = unsafe { libc::poll(&mut fd, 1, timeout) };
            if ready > 0 {
                tolerate_would_block(guard.read())?;
            } else if ready < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() != std::io::ErrorKind::Interrupted {
                    return Err(error.into());
                }
            }
        }
    }

    Ok(())
}

// Read readiness can be spurious; Wayland uses a nonblocking socket.
fn tolerate_would_block<T>(
    result: Result<T, wayland_client::backend::WaylandError>,
) -> Result<(), OverlayError> {
    match result {
        Ok(_) => Ok(()),
        Err(wayland_client::backend::WaylandError::Io(error))
            if error.kind() == std::io::ErrorKind::WouldBlock =>
        {
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn create_overlay_layer(
    compositor: &CompositorState,
    layer_shell: &LayerShell,
    viewporter: Option<&WpViewporter>,
    qh: &QueueHandle<Overlay>,
    width: u32,
    height: u32,
    output: Option<&wl_output::WlOutput>,
) -> (LayerSurface, Option<WpViewport>) {
    let surface = compositor.create_surface(qh);
    let layer =
        layer_shell.create_layer_surface(qh, surface, Layer::Overlay, Some("whisrs"), output);
    layer.set_anchor(Anchor::BOTTOM);
    layer.set_margin(0, 0, BOTTOM_MARGIN, 0);
    layer.set_exclusive_zone(0);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.set_size(width, height);

    let viewport = viewporter.map(|viewporter| {
        let viewport = viewporter.get_viewport(layer.wl_surface(), qh, ());
        viewport.set_destination(width as i32, height as i32);
        viewport
    });

    // Make the transparent overlay non-interactive so it never blocks clicks.
    let input_region = compositor.wl_compositor().create_region(qh, ());
    layer.set_input_region(Some(&input_region));
    input_region.destroy();
    layer.commit();
    (layer, viewport)
}

fn run_x11_overlay(
    state_rx: mpsc::Receiver<State>,
    level_rx: mpsc::Receiver<f32>,
    config: OverlayConfig,
) -> Result<(), OverlayError> {
    let width = config.clamped_width() as u16;
    let height = config.clamped_height() as u16;
    let sensitivity = config.clamped_sensitivity();
    let theme = Theme::from_config(&config);

    let (conn, screen_num) = RustConnection::connect(None)?;
    let screen = &conn.setup().roots[screen_num];
    let visual = find_argb_visual(screen).ok_or(OverlayError::X11Visual)?;
    let atoms = X11Atoms::new(&conn)?;

    let colormap = conn.generate_id()?;
    conn.create_colormap(ColormapAlloc::NONE, colormap, screen.root, visual.visual_id)?;

    let x = centered_x(screen, width);
    let y = bottom_y(screen, height);
    let window = conn.generate_id()?;
    conn.create_window(
        visual.depth,
        window,
        screen.root,
        x,
        y,
        width,
        height,
        0,
        WindowClass::INPUT_OUTPUT,
        visual.visual_id,
        &CreateWindowAux::default()
            .background_pixel(0)
            .border_pixel(0)
            .colormap(colormap)
            .override_redirect(1)
            .event_mask(EventMask::EXPOSURE),
    )?;

    set_x11_window_hints(&conn, window, &atoms)?;
    let shape_supported = make_x11_window_click_through(&conn, window)?;

    let gc = conn.generate_id()?;
    conn.create_gc(gc, window, &CreateGCAux::default().graphics_exposures(0))?;
    conn.flush()?;

    let mut renderer = OverlayRenderer::new(
        state_rx,
        level_rx,
        width as u32,
        height as u32,
        theme,
        sensitivity,
    )?;
    let mut frame = vec![0_u8; width as usize * height as usize * 4];
    let mut mapped = false;
    // Cache of the last bounding-shape rectangles applied to the window.
    // Recomputing alpha_shape_rectangles every frame is cheap, but issuing
    // the SHAPE round-trip when geometry hasn't changed is not — skip it.
    let mut last_shape: Vec<Rectangle> = Vec::new();

    info!("recording X11 overlay started");
    loop {
        // Drain the event queue. Redraws are time-driven (the loop below
        // pushes a frame every `FRAME_MS`), so we don't react to `Expose`
        // explicitly — we just consume events so the protocol stream
        // stays well-formed.
        while conn.poll_for_event()?.is_some() {}

        renderer.draw_frame();
        if renderer.disconnected {
            break;
        }
        let visible_shape = alpha_shape_rectangles(&renderer.pixmap);
        if visible_shape.is_empty() {
            if mapped {
                conn.unmap_window(window)?;
                conn.flush()?;
                mapped = false;
                last_shape.clear();
            }
            std::thread::sleep(Duration::from_millis(FRAME_MS));
            continue;
        }

        if shape_supported && !shape_rectangles_eq(&visible_shape, &last_shape) {
            set_x11_bounding_shape(&conn, window, &visible_shape)?;
            last_shape.clear();
            last_shape.extend_from_slice(&visible_shape);
        }
        copy_pixmap_to_x11_zpixmap(&renderer.pixmap, visual, &mut frame);
        if !mapped {
            let (x, y) = x11_overlay_position(&conn, screen, &atoms, width, height);
            conn.configure_window(
                window,
                &ConfigureWindowAux::default()
                    .x(i32::from(x))
                    .y(i32::from(y))
                    .stack_mode(StackMode::ABOVE),
            )?;
            conn.map_window(window)?;
            mapped = true;
        }
        conn.put_image(
            ImageFormat::Z_PIXMAP,
            window,
            gc,
            width,
            height,
            0,
            0,
            0,
            visual.depth,
            &frame,
        )?;
        conn.flush()?;
        std::thread::sleep(Duration::from_millis(FRAME_MS));
    }

    if mapped {
        let _ = conn.unmap_window(window);
        let _ = conn.flush();
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct X11ArgbVisual {
    depth: u8,
    visual_id: Visualid,
    red_mask: u32,
    green_mask: u32,
    blue_mask: u32,
    alpha_mask: u32,
}

fn find_argb_visual(screen: &Screen) -> Option<X11ArgbVisual> {
    screen.allowed_depths.iter().find_map(|depth| {
        if depth.depth != 32 {
            return None;
        }
        let full_mask = depth_mask(depth.depth);
        depth.visuals.iter().find_map(|visual| {
            if visual.class != VisualClass::TRUE_COLOR {
                return None;
            }
            let color_mask = visual.red_mask | visual.green_mask | visual.blue_mask;
            let alpha_mask = full_mask & !color_mask;
            (alpha_mask != 0).then_some(X11ArgbVisual {
                depth: depth.depth,
                visual_id: visual.visual_id,
                red_mask: visual.red_mask,
                green_mask: visual.green_mask,
                blue_mask: visual.blue_mask,
                alpha_mask,
            })
        })
    })
}

fn depth_mask(depth: u8) -> u32 {
    if depth >= 32 {
        u32::MAX
    } else {
        (1_u32 << depth) - 1
    }
}

struct X11Atoms {
    atom: u32,
    net_active_window: u32,
    net_wm_name: u32,
    net_wm_window_type: u32,
    net_wm_window_type_notification: u32,
    net_wm_state: u32,
    net_wm_state_above: u32,
    net_wm_state_skip_pager: u32,
    net_wm_state_skip_taskbar: u32,
    net_wm_state_sticky: u32,
    net_wm_bypass_compositor: u32,
    utf8_string: u32,
}

impl X11Atoms {
    fn new(conn: &RustConnection) -> Result<Self, OverlayError> {
        Ok(Self {
            atom: u32::from(AtomEnum::ATOM),
            net_active_window: intern_atom(conn, b"_NET_ACTIVE_WINDOW")?,
            net_wm_name: intern_atom(conn, b"_NET_WM_NAME")?,
            net_wm_window_type: intern_atom(conn, b"_NET_WM_WINDOW_TYPE")?,
            net_wm_window_type_notification: intern_atom(
                conn,
                b"_NET_WM_WINDOW_TYPE_NOTIFICATION",
            )?,
            net_wm_state: intern_atom(conn, b"_NET_WM_STATE")?,
            net_wm_state_above: intern_atom(conn, b"_NET_WM_STATE_ABOVE")?,
            net_wm_state_skip_pager: intern_atom(conn, b"_NET_WM_STATE_SKIP_PAGER")?,
            net_wm_state_skip_taskbar: intern_atom(conn, b"_NET_WM_STATE_SKIP_TASKBAR")?,
            net_wm_state_sticky: intern_atom(conn, b"_NET_WM_STATE_STICKY")?,
            net_wm_bypass_compositor: intern_atom(conn, b"_NET_WM_BYPASS_COMPOSITOR")?,
            utf8_string: intern_atom(conn, b"UTF8_STRING")?,
        })
    }
}

fn intern_atom(conn: &RustConnection, name: &[u8]) -> Result<u32, OverlayError> {
    Ok(conn.intern_atom(false, name)?.reply()?.atom)
}

fn set_x11_window_hints(
    conn: &RustConnection,
    window: Window,
    atoms: &X11Atoms,
) -> Result<(), OverlayError> {
    // Override-redirect windows normally avoid window-manager focus handling,
    // but explicitly mark this HUD as non-input as well. XWayland compositors
    // and Electron clients can otherwise react to the overlay being mapped by
    // dropping focus from an embedded text field even though the application
    // window itself remains active.
    let mut wm_hints = WmHints::new();
    wm_hints.input = Some(false);
    wm_hints.set(conn, window)?;

    conn.change_property8(
        PropMode::REPLACE,
        window,
        atoms.net_wm_name,
        atoms.utf8_string,
        b"whisrs overlay",
    )?;
    conn.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"whisrs overlay",
    )?;
    conn.change_property32(
        PropMode::REPLACE,
        window,
        atoms.net_wm_window_type,
        atoms.atom,
        &[atoms.net_wm_window_type_notification],
    )?;
    conn.change_property32(
        PropMode::REPLACE,
        window,
        atoms.net_wm_state,
        atoms.atom,
        &[
            atoms.net_wm_state_above,
            atoms.net_wm_state_skip_pager,
            atoms.net_wm_state_skip_taskbar,
            atoms.net_wm_state_sticky,
        ],
    )?;
    conn.change_property32(
        PropMode::REPLACE,
        window,
        atoms.net_wm_bypass_compositor,
        AtomEnum::CARDINAL,
        &[2],
    )?;
    Ok(())
}

fn make_x11_window_click_through(
    conn: &RustConnection,
    window: Window,
) -> Result<bool, OverlayError> {
    if conn.extension_information(SHAPE_EXT)?.is_none() {
        warn!("X11 SHAPE extension unavailable; overlay may intercept clicks");
        return Ok(false);
    }

    conn.shape_rectangles(
        SO::SET,
        SK::INPUT,
        ClipOrdering::UNSORTED,
        window,
        0,
        0,
        &[],
    )?;
    Ok(true)
}

fn set_x11_bounding_shape(
    conn: &RustConnection,
    window: Window,
    rectangles: &[Rectangle],
) -> Result<(), OverlayError> {
    conn.shape_rectangles(
        SO::SET,
        SK::BOUNDING,
        ClipOrdering::Y_SORTED,
        window,
        0,
        0,
        rectangles,
    )?;
    Ok(())
}

fn shape_rectangles_eq(a: &[Rectangle], b: &[Rectangle]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(lhs, rhs)| {
        lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width && lhs.height == rhs.height
    })
}

fn alpha_shape_rectangles(pixmap: &Pixmap) -> Vec<Rectangle> {
    let width = pixmap.width() as usize;
    let mut rectangles = Vec::new();

    for (y, row) in pixmap.pixels().chunks_exact(width).enumerate() {
        let Some(first) = row.iter().position(|px| px.alpha() != 0) else {
            continue;
        };
        let last = row
            .iter()
            .rposition(|px| px.alpha() != 0)
            .expect("row has a first alpha pixel");
        rectangles.push(Rectangle {
            x: first as i16,
            y: y as i16,
            width: (last - first + 1) as u16,
            height: 1,
        });
    }

    rectangles
}

fn x11_overlay_position(
    conn: &RustConnection,
    screen: &Screen,
    atoms: &X11Atoms,
    width: u16,
    height: u16,
) -> (i16, i16) {
    let monitors = match active_x11_monitors(conn, screen.root) {
        Ok(monitors) => monitors,
        Err(e) => {
            warn!("failed to query XRandR monitors for overlay placement: {e:#}");
            Vec::new()
        }
    };

    if let Some((cx, cy)) = match active_x11_window_center(conn, screen.root, atoms) {
        Ok(center) => center,
        Err(e) => {
            warn!("failed to query active X11 window for overlay placement: {e:#}");
            None
        }
    } {
        if let Some(monitor) = monitors
            .iter()
            .find(|m| point_in_rect(cx, cy, i32::from(m.x), i32::from(m.y), m.width, m.height))
        {
            return position_in_rect(
                i32::from(monitor.x),
                i32::from(monitor.y),
                u32::from(monitor.width),
                u32::from(monitor.height),
                width,
                height,
            );
        }
    }

    if let Some(monitor) = monitors
        .iter()
        .find(|m| m.primary)
        .or_else(|| monitors.first())
    {
        return position_in_rect(
            i32::from(monitor.x),
            i32::from(monitor.y),
            u32::from(monitor.width),
            u32::from(monitor.height),
            width,
            height,
        );
    }

    (centered_x(screen, width), bottom_y(screen, height))
}

fn active_x11_monitors(
    conn: &RustConnection,
    root: Window,
) -> Result<Vec<MonitorInfo>, OverlayError> {
    if conn.extension_information(RANDR_EXT)?.is_none() {
        return Ok(Vec::new());
    }

    let reply = conn.randr_get_monitors(root, true)?.reply()?;
    Ok(reply
        .monitors
        .into_iter()
        .filter(|monitor| monitor.width > 0 && monitor.height > 0)
        .collect())
}

fn active_x11_window_center(
    conn: &RustConnection,
    root: Window,
    atoms: &X11Atoms,
) -> Result<Option<(i32, i32)>, OverlayError> {
    let reply = conn
        .get_property(false, root, atoms.net_active_window, AtomEnum::WINDOW, 0, 1)?
        .reply()?;

    let Some(bytes) = reply.value.get(..4) else {
        return Ok(None);
    };
    let window = u32::from_ne_bytes(bytes.try_into().expect("slice is exactly 4 bytes"));
    if window == 0 {
        return Ok(None);
    }

    let geometry = conn.get_geometry(window)?.reply()?;
    let translated = conn.translate_coordinates(window, root, 0, 0)?.reply()?;
    if !translated.same_screen {
        return Ok(None);
    }

    Ok(Some((
        i32::from(translated.dst_x) + i32::from(geometry.width / 2),
        i32::from(translated.dst_y) + i32::from(geometry.height / 2),
    )))
}

fn point_in_rect(x: i32, y: i32, rx: i32, ry: i32, width: u16, height: u16) -> bool {
    let right = rx.saturating_add(i32::from(width));
    let bottom = ry.saturating_add(i32::from(height));
    x >= rx && x < right && y >= ry && y < bottom
}

fn position_in_rect(
    x: i32,
    y: i32,
    rect_width: u32,
    rect_height: u32,
    width: u16,
    height: u16,
) -> (i16, i16) {
    let width = i32::from(width);
    let height = i32::from(height);
    let rect_width = i32::try_from(rect_width).unwrap_or(i32::MAX);
    let rect_height = i32::try_from(rect_height).unwrap_or(i32::MAX);

    let min_x = x;
    let max_x = x
        .saturating_add(rect_width)
        .saturating_sub(width)
        .max(min_x);
    let centered_x = x.saturating_add((rect_width - width) / 2);

    let min_y = y;
    let max_y = y
        .saturating_add(rect_height)
        .saturating_sub(height)
        .max(min_y);
    let bottom_y = y
        .saturating_add(rect_height)
        .saturating_sub(height)
        .saturating_sub(BOTTOM_MARGIN);

    (
        to_i16_coord(centered_x.clamp(min_x, max_x)),
        to_i16_coord(bottom_y.clamp(min_y, max_y)),
    )
}

fn centered_x(screen: &Screen, width: u16) -> i16 {
    let x = (i32::from(screen.width_in_pixels) - i32::from(width)) / 2;
    to_i16_coord(x.max(0))
}

fn bottom_y(screen: &Screen, height: u16) -> i16 {
    let y = i32::from(screen.height_in_pixels) - i32::from(height) - BOTTOM_MARGIN;
    to_i16_coord(y.max(0))
}

fn to_i16_coord(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

struct Overlay {
    registry_state: RegistryState,
    output_state: OutputState,
    compositor: CompositorState,
    layer_shell: LayerShell,
    viewporter: Option<WpViewporter>,
    viewport: Option<WpViewport>,
    shm: Shm,
    pool: SlotPool,
    layer: LayerSurface,
    focused_output_rx: mpsc::Receiver<String>,
    desired_output_name: Option<String>,
    current_output_name: Option<String>,
    renderer: OverlayRenderer,
    logical_width: u32,
    logical_height: u32,
    exit: bool,
    first_configure: bool,
}

struct OverlayRenderer {
    state_rx: mpsc::Receiver<State>,
    level_rx: mpsc::Receiver<f32>,
    width: u32,
    height: u32,
    scale: f32,
    pixmap: Pixmap,
    target_state: State,
    visible_state: State,
    /// Wall-clock instant when the current spawn animation started. The
    /// animation progress `t` is derived from `(now - spawn_started) /
    /// duration`, so the timing is honest regardless of how often the
    /// dispatch loop ticks. (Previously this was a per-call increment that
    /// over-advanced when the loop fired multiple times per rendered
    /// frame, making the animation visually pop instead of ease.)
    spawn_started: Instant,
    /// `true` while transitioning into a visible state, `false` while
    /// transitioning out. Determines easing direction and duration.
    spawn_in: bool,
    /// Set once either input channel returns `Disconnected` — the daemon's
    /// forwarding task has exited, so the overlay should wind down cleanly
    /// instead of leaking the thread.
    disconnected: bool,
    frame: u32,
    /// Smoothed audio level driving bar heights. Advanced toward
    /// `level_target` by a critically-damped spring stepped with the
    /// real elapsed `dt` between calls — frame-rate-independent.
    level: f32,
    level_target: f32,
    level_velocity: f32,
    /// Wall-clock instant of the previous spring step.
    last_update: Instant,
    theme: Theme,
    sensitivity: f32,
}

/// Per-frame animation state computed from `spawn_t` + `spawn_in`. The
/// spawn animation drives a bottom-anchored height morph (with a small
/// overshoot on appear) instead of a slide+scale; the pill literally draws
/// itself out of the screen edge.
#[derive(Debug, Clone, Copy)]
struct AnimState {
    /// Currently displayed pill height in px (bottom-anchored — the pill's
    /// bottom edge stays glued to the surface bottom regardless of this
    /// value).
    pill_height: f32,
    /// Pill alpha 0..=1. Eases in faster than the height grows so the pill
    /// is solid before it stops moving.
    pill_alpha: f32,
    /// Bar alpha 0..=1. Stays at 0 during the initial grow, then fades in.
    bar_alpha: f32,
    /// `true` while audio reactivity should be gated to baseline. Honored
    /// by the recording draw — silenced bars rise from baseline once this
    /// goes false.
    bars_locked: bool,
}

impl OverlayRenderer {
    fn new(
        state_rx: mpsc::Receiver<State>,
        level_rx: mpsc::Receiver<f32>,
        width: u32,
        height: u32,
        theme: Theme,
        sensitivity: f32,
    ) -> Result<Self, OverlayError> {
        let pixmap = Pixmap::new(width, height).ok_or(OverlayError::Pixmap(width, height))?;
        Ok(Self {
            state_rx,
            level_rx,
            width,
            height,
            scale: 1.0,
            pixmap,
            target_state: State::Idle,
            visible_state: State::Idle,
            spawn_started: Instant::now(),
            spawn_in: false,
            disconnected: false,
            frame: 0,
            level: 0.0,
            level_target: 0.0,
            level_velocity: 0.0,
            last_update: Instant::now(),
            theme,
            sensitivity,
        })
    }

    fn resize_for_scale(
        &mut self,
        logical_width: u32,
        logical_height: u32,
        scale: u32,
    ) -> Result<(), OverlayError> {
        let scale = scale.max(1);
        self.scale = scale as f32;
        self.width = logical_width.saturating_mul(scale);
        self.height = logical_height.saturating_mul(scale);
        self.pixmap = Pixmap::new(self.width, self.height)
            .ok_or(OverlayError::Pixmap(self.width, self.height))?;
        Ok(())
    }

    fn apply_state_updates(&mut self) {
        loop {
            match self.state_rx.try_recv() {
                Ok(state) => {
                    let was_idle = self.target_state == State::Idle;
                    let now_idle = state == State::Idle;
                    self.target_state = state;

                    // Completion must clear on the next frame, without a
                    // trailing disappearance animation after text is pasted.
                    self.visible_state = state;

                    // Trigger spawn / despawn only on the boundary between
                    // idle and visible. Recording ↔ Transcribing keeps the
                    // pill steady.
                    if was_idle && !now_idle {
                        self.spawn_in = true;
                        self.spawn_started = Instant::now();
                    } else if !was_idle && now_idle {
                        self.spawn_in = false;
                        self.spawn_started = Instant::now();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.disconnected = true;
                    break;
                }
            }
        }
        // Drain incoming audio levels — keep only the latest as the
        // spring target. The spring is stepped below using real elapsed
        // `dt`, so it doesn't matter how many samples we drained.
        loop {
            match self.level_rx.try_recv() {
                Ok(new) => self.level_target = (new * self.sensitivity).clamp(0.0, 1.0),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.disconnected = true;
                    break;
                }
            }
        }

        // Critically-damped-ish spring on the displayed level. Stepped
        // with wall-clock dt so the time constants are real-world ms,
        // not "per dispatch tick". `STIFFNESS` and `DAMPING` are tuned
        // to settle in ~150–200 ms with no perceptible overshoot — feels
        // like the bars *track* the voice rather than chase it.
        const STIFFNESS: f32 = 360.0;
        const DAMPING: f32 = 32.0;
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32().min(0.1);
        self.last_update = now;
        if dt > 0.0 {
            let force = (self.level_target - self.level) * STIFFNESS;
            let drag = self.level_velocity * DAMPING;
            self.level_velocity += (force - drag) * dt;
            self.level = (self.level + self.level_velocity * dt).clamp(0.0, 1.0);
        }

        // Once the despawn finishes, snap the renderer to Idle so the next
        // appearance starts from a clean state.
        if !self.spawn_in && self.spawn_t() >= 1.0 {
            self.visible_state = State::Idle;
        }
    }

    /// Wall-clock progress through the current spawn animation, 0..=1.
    fn spawn_t(&self) -> f32 {
        let duration = if self.spawn_in {
            SPAWN_IN_MS
        } else {
            SPAWN_OUT_MS
        };
        let elapsed_ms = self.spawn_started.elapsed().as_secs_f32() * 1000.0;
        (elapsed_ms / duration).clamp(0.0, 1.0)
    }

    /// Compute the current animated transform.
    ///
    /// **Appear** (`spawn_in == true`, 220 ms):
    /// - Pill height grows `SPAWN_PILL_MIN_H → full_height` with an
    ///   `easeOutBack` curve, so it overshoots ~3 % then settles —
    ///   gives the arrival a small "physical pop".
    /// - Pill alpha eases in over the first ~64 % of the duration (so
    ///   the pill is solid before it stops moving — Material 3
    ///   "emphasized" pattern).
    /// - Bars stay invisible for the first 80 ms, then fade in over the
    ///   next 80 ms while the pill is still finishing its grow. After
    ///   that window they unlock and react to audio.
    ///
    /// **Disappear** (`spawn_in == false`, 140 ms):
    /// - Pill height shrinks back to `SPAWN_PILL_MIN_H` with an
    ///   `easeInCubic` accelerate — sharper exit than entry.
    /// - Pill and bar alpha fade in lockstep with the height collapse.
    fn anim(&self, full_height: f32) -> AnimState {
        let t = self.spawn_t();
        if self.spawn_in {
            let h_curve = ease_out_back(t, SPAWN_OVERSHOOT_C).clamp(0.0, 1.4);
            let pill_height = SPAWN_PILL_MIN_H + h_curve * (full_height - SPAWN_PILL_MIN_H);

            // Alpha finishes well before the height — so the pill is fully
            // opaque while it's still growing. ease-out-quad over first 64%.
            let alpha_t = (t / 0.64).clamp(0.0, 1.0);
            let pill_alpha = 1.0 - (1.0 - alpha_t) * (1.0 - alpha_t);

            // Bar grace + fade. Convert ms to t-fractions of total duration.
            let grace_t = BARS_GRACE_MS / SPAWN_IN_MS;
            let fade_t = BARS_FADE_MS / SPAWN_IN_MS;
            let bar_t = ((t - grace_t) / fade_t).clamp(0.0, 1.0);
            let bar_alpha = 1.0 - (1.0 - bar_t) * (1.0 - bar_t);
            let bars_locked = t < grace_t + fade_t;

            AnimState {
                pill_height,
                pill_alpha,
                bar_alpha,
                bars_locked,
            }
        } else {
            let e = ease_in_cubic(t);
            let pill_height = full_height - e * (full_height - SPAWN_PILL_MIN_H);
            let pill_alpha = 1.0 - e;
            // Bars fade out a bit faster than the pill collapses.
            let bar_t = (t / 0.7).clamp(0.0, 1.0);
            let bar_alpha = 1.0 - bar_t * bar_t;
            AnimState {
                pill_height,
                pill_alpha,
                bar_alpha,
                bars_locked: true,
            }
        }
    }

    fn draw_frame(&mut self) {
        self.apply_state_updates();

        let mut anim = self.anim(self.height as f32);
        anim.pill_height = anim.pill_height.max(SPAWN_PILL_MIN_H * self.scale);
        let level_gated = if anim.bars_locked { 0.0 } else { self.level };
        draw_overlay(
            &mut self.pixmap,
            self.visible_state,
            self.frame,
            level_gated,
            anim,
            &self.theme,
            self.scale,
        );
        self.frame = self.frame.wrapping_add(1);
    }
}

impl Overlay {
    fn apply_focused_output(&mut self, qh: &QueueHandle<Self>) {
        while let Ok(name) = self.focused_output_rx.try_recv() {
            self.desired_output_name = Some(name);
        }

        let Some(desired_name) = self.desired_output_name.clone() else {
            return;
        };
        if self.current_output_name.as_deref() == Some(&desired_name) {
            return;
        }

        let target_output = self.output_state.outputs().find(|output| {
            self.output_state
                .info(output)
                .and_then(|info| info.name)
                .as_deref()
                == Some(desired_name.as_str())
        });
        let Some(target_output) = target_output else {
            return;
        };

        let (new_layer, new_viewport) = create_overlay_layer(
            &self.compositor,
            &self.layer_shell,
            self.viewporter.as_ref(),
            qh,
            self.logical_width,
            self.logical_height,
            Some(&target_output),
        );
        if let Some(viewport) = self.viewport.take() {
            viewport.destroy();
        }
        self.layer = new_layer;
        self.viewport = new_viewport;
        self.current_output_name = Some(desired_name.clone());
        self.first_configure = true;
        info!("recording overlay moved to output {desired_name}");
    }

    fn draw(&mut self, _qh: &QueueHandle<Self>) {
        let width = self.renderer.width;
        let height = self.renderer.height;
        let stride = width as i32 * 4;

        self.renderer.draw_frame();

        let Ok((buffer, canvas)) = self.pool.create_buffer(
            width as i32,
            height as i32,
            stride,
            wl_shm::Format::Argb8888,
        ) else {
            warn!("failed to allocate overlay buffer");
            return;
        };
        copy_pixmap_to_argb8888(&self.renderer.pixmap, canvas);

        self.layer
            .wl_surface()
            .damage_buffer(0, 0, width as i32, height as i32);
        if let Err(e) = buffer.attach_to(self.layer.wl_surface()) {
            warn!("failed to attach overlay buffer: {e}");
            return;
        }
        self.layer.commit();
    }
}

fn ease_in_cubic(t: f32) -> f32 {
    t * t * t
}

/// "Back" easing — overshoots the target by `c × peak %` before settling.
/// `c = 0.4` ⇒ ~3 % overshoot; `c = 1.7` is the standard CSS easeOutBack
/// (~10 %). Output range is approximately [0, 1+c/something], typically
/// peaking around `t ≈ 0.85`.
fn ease_out_back(t: f32, c: f32) -> f32 {
    let t1 = t - 1.0;
    1.0 + (c + 1.0) * t1 * t1 * t1 + c * t1 * t1
}

/// tiny-skia stores premultiplied RGBA bytes; the wl_shm Argb8888 format on
/// little-endian systems is BGRA in memory. Convert by swapping R and B.
/// Both formats use premultiplied alpha so no math is needed beyond the swap.
fn copy_pixmap_to_argb8888(pixmap: &Pixmap, canvas: &mut [u8]) {
    let src = pixmap.pixels();
    debug_assert_eq!(src.len() * 4, canvas.len());
    for (i, px) in src.iter().enumerate() {
        let dst = &mut canvas[i * 4..i * 4 + 4];
        let pre: PremultipliedColorU8 = *px;
        dst[0] = pre.blue();
        dst[1] = pre.green();
        dst[2] = pre.red();
        dst[3] = pre.alpha();
    }
}

fn copy_pixmap_to_x11_zpixmap(pixmap: &Pixmap, visual: X11ArgbVisual, canvas: &mut [u8]) {
    let src = pixmap.pixels();
    debug_assert_eq!(src.len() * 4, canvas.len());
    for (i, px) in src.iter().enumerate() {
        let pre: PremultipliedColorU8 = *px;
        let pixel = pack_masked_channel(pre.red(), visual.red_mask)
            | pack_masked_channel(pre.green(), visual.green_mask)
            | pack_masked_channel(pre.blue(), visual.blue_mask)
            | pack_masked_channel(pre.alpha(), visual.alpha_mask);
        canvas[i * 4..i * 4 + 4].copy_from_slice(&pixel.to_ne_bytes());
    }
}

/// Pack an 8-bit channel `value` into the bits described by `mask`, scaling
/// it from `[0, 255]` into `[0, mask_max]`. The `(value * max + 127) / 255`
/// form is the textbook integer rescale: adding `127` before dividing by
/// `255` performs round-half-up so the brightest input (`0xFF`) always
/// produces the brightest output, and intermediate values are nearest-
/// rounded instead of truncated toward zero.
fn pack_masked_channel(value: u8, mask: u32) -> u32 {
    if mask == 0 {
        return 0;
    }
    let bits = mask.count_ones();
    let shift = mask.trailing_zeros();
    let max = (1_u64 << bits) - 1;
    let scaled = (u64::from(value) * max + 127) / 255;
    ((scaled << shift) as u32) & mask
}

impl CompositorHandler for Overlay {
    fn scale_factor_changed(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        if surface != self.layer.wl_surface() || self.viewport.is_none() {
            return;
        }
        let scale = new_factor.max(1) as u32;
        if scale == self.renderer.scale as u32 {
            return;
        }
        if let Err(error) =
            self.renderer
                .resize_for_scale(self.logical_width, self.logical_height, scale)
        {
            warn!("failed to resize overlay for output scale {scale}: {error}");
            return;
        }
        match SlotPool::new(
            (self.renderer.width * self.renderer.height * 4) as usize,
            &self.shm,
        ) {
            Ok(pool) => self.pool = pool,
            Err(error) => warn!("failed to resize overlay buffer pool: {error}"),
        }
    }

    fn transform_changed(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &WaylandConnection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
        if surface != self.layer.wl_surface() {
            return;
        }
        // Rendering is driven by the bounded event loop.
        let _ = qh;
    }

    fn surface_enter(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for Overlay {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn update_output(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for Overlay {
    fn closed(&mut self, _conn: &WaylandConnection, _qh: &QueueHandle<Self>, layer: &LayerSurface) {
        if layer == &self.layer {
            self.exit = true;
        }
    }

    fn configure(
        &mut self,
        _conn: &WaylandConnection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        _configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        if layer != &self.layer {
            return;
        }
        if self.first_configure {
            self.first_configure = false;
            self.draw(qh);
        }
    }
}

impl ShmHandler for Overlay {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl Dispatch<WpViewporter, ()> for Overlay {
    fn event(
        _state: &mut Self,
        _proxy: &WpViewporter,
        _event: wp_viewporter::Event,
        _data: &(),
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
    ) {
        unreachable!("wp_viewporter has no events")
    }
}

impl Dispatch<WpViewport, ()> for Overlay {
    fn event(
        _state: &mut Self,
        _proxy: &WpViewport,
        _event: wp_viewport::Event,
        _data: &(),
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
    ) {
        unreachable!("wp_viewport has no events")
    }
}

impl Dispatch<wl_region::WlRegion, ()> for Overlay {
    fn event(
        _state: &mut Self,
        _proxy: &wl_region::WlRegion,
        _event: wl_region::Event,
        _data: &(),
        _conn: &WaylandConnection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

delegate_compositor!(Overlay);
delegate_output!(Overlay);
delegate_shm!(Overlay);
delegate_layer!(Overlay);
delegate_registry!(Overlay);

impl ProvidesRegistryState for Overlay {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers![OutputState];
}

/// Render one frame of the overlay into `pixmap` using tiny-skia. The
/// pixmap is the same size as the surface; the pill is drawn at the bottom
/// of the surface (its bottom edge glued to the surface bottom) so the
/// height-morph animation reads as the pill *growing out of the screen
/// edge* instead of inflating from its center.
fn draw_overlay(
    pixmap: &mut Pixmap,
    state: State,
    frame: u32,
    level: f32,
    anim: AnimState,
    theme: &Theme,
    scale: f32,
) {
    pixmap.fill(Color::TRANSPARENT);

    if anim.pill_alpha <= 0.0 || state == State::Idle {
        return;
    }

    let surface_w = pixmap.width() as f32;
    let surface_h = pixmap.height() as f32;
    let pill_h = anim.pill_height.clamp(SPAWN_PILL_MIN_H * scale, surface_h);
    let pill_y = surface_h - pill_h; // bottom-anchored
    let pill_w = surface_w;

    // Outer pill — drawn in the *ring* color first. We then paint a 1 px
    // inset pill in the *bg* color so the visible result is a perfectly
    // even 1 px ring with no stroke-math seams. This avoids the corner
    // join artifacts that a tiny-skia stroke produces on tight pills.
    let outer = build_stadium(0.0, pill_y, pill_w, pill_h);
    if let Some(path) = &outer {
        let mut paint = Paint {
            anti_alias: true,
            ..Default::default()
        };
        paint.set_color(theme_color(theme.ring, anim.pill_alpha));
        pixmap.fill_path(path, &paint, FillRule::Winding, Transform::identity(), None);
    }

    let ring = scale;
    if pill_w > ring * 2.0 && pill_h > ring * 2.0 {
        let inner = build_stadium(
            ring,
            pill_y + ring,
            pill_w - ring * 2.0,
            pill_h - ring * 2.0,
        );
        if let Some(path) = &inner {
            let mut paint = Paint {
                anti_alias: true,
                ..Default::default()
            };
            paint.set_color(theme_color(theme.bg, anim.pill_alpha));
            pixmap.fill_path(path, &paint, FillRule::Winding, Transform::identity(), None);
        }
    }

    if anim.bar_alpha <= 0.0 {
        return;
    }

    // Bars sit at the *final* pill center — the surface midpoint — not at
    // the currently animating pill_cy. While the pill is still growing
    // upward from the bottom edge, this keeps the bars planted at one fixed
    // y so they read as "expanding amplitude" rather than "translating up
    // with the pill". The pill's bottom-anchored grow still happens
    // visually; the bars just don't follow its center-of-mass.
    let pill_cy = surface_h / 2.0;
    match state {
        State::Recording => draw_bars(pixmap, theme, theme.rec_bar, level, anim, pill_cy, scale),
        // Read-aloud playback reuses the equalizer bars in a distinct hue,
        // driven by the spoken audio amplitude.
        State::Speaking => draw_bars(pixmap, theme, theme.speak_bar, level, anim, pill_cy, scale),
        // Synthesizing reuses the transcribing "working on it" shimmer.
        State::Transcribing | State::Synthesizing => {
            draw_sweep(pixmap, theme, frame, anim, pill_cy, scale)
        }
        State::Idle => {}
    }
    // NOTE: the external GNOME Shell extension (not in this repo) only renders
    // recording/transcribing today; it would need a separate update to draw
    // the synthesizing/speaking states. The native Wayland/X11 paths above
    // handle them fully.
}

/// Build a stadium / pill path: rectangle + two end-cap circles, joined by
/// the `Winding` fill rule. This is geometrically exact — no cubic-bezier
/// approximation — so the silhouette has no minor inward dents at the
/// corner joins, which were visible against colored rings.
///
/// Handles both orientations: horizontal pills (`w > h`) get end caps on
/// the left/right; vertical pills (`h > w`) get end caps on the top/bottom.
/// Square (or near-square) input collapses to a single circle. Returns
/// `None` for non-positive dimensions.
fn build_stadium(x: f32, y: f32, w: f32, h: f32) -> Option<tiny_skia::Path> {
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    if w >= h {
        let r = h / 2.0;
        if (w - 2.0 * r).abs() < 0.01 {
            return PathBuilder::from_circle(x + r, y + r, r);
        }
        let mut pb = PathBuilder::new();
        if let Some(rect) = Rect::from_xywh(x + r, y, w - 2.0 * r, h) {
            pb.push_rect(rect);
        }
        if let Some(cap) = PathBuilder::from_circle(x + r, y + r, r) {
            pb.push_path(&cap);
        }
        if let Some(cap) = PathBuilder::from_circle(x + w - r, y + r, r) {
            pb.push_path(&cap);
        }
        pb.finish()
    } else {
        let r = w / 2.0;
        let mut pb = PathBuilder::new();
        if let Some(rect) = Rect::from_xywh(x, y + r, w, h - 2.0 * r) {
            pb.push_rect(rect);
        }
        if let Some(cap) = PathBuilder::from_circle(x + r, y + r, r) {
            pb.push_path(&cap);
        }
        if let Some(cap) = PathBuilder::from_circle(x + r, y + h - r, r) {
            pb.push_path(&cap);
        }
        pb.finish()
    }
}

/// Convert a `[A, R, G, B]` byte array to a tiny-skia non-premultiplied
/// `Color`, with the alpha channel further scaled by `extra_alpha`.
fn theme_color(bytes: [u8; 4], extra_alpha: f32) -> Color {
    let a = (bytes[0] as f32 / 255.0 * extra_alpha.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    Color::from_rgba(
        bytes[1] as f32 / 255.0,
        bytes[2] as f32 / 255.0,
        bytes[3] as f32 / 255.0,
        a,
    )
    .unwrap_or(Color::TRANSPARENT)
}

/// Wavy taper across the bar row — center bar at ~100 %, with a cosine
/// modulation so adjacent bars alternate between "taller" and "shorter"
/// inside a gaussian envelope. Reads as an equalizer pattern instead of
/// a smooth bell:
///
/// ```text
///   index:    0     1     2     3     4     5     6
///   factor:  .20   .64   .45  1.00   .45   .64   .20
///            short  tall  mid  PEAK  mid  tall  short
/// ```
fn taper_factor(i: u32, count: u32) -> f32 {
    if count <= 1 {
        return 1.0;
    }
    let center = (count as f32 - 1.0) / 2.0;
    let d = (i as f32 - center) / center; // -1..=1
    let envelope = (-d * d).exp(); // exp(-1) ≈ 0.367 at edges
                                   // For odd `count`, (i - center) is integer ⇒ cos is ±1, giving a
                                   // strict alternation. For even `count` the cos collapses to 0 and the
                                   // factor is just the gaussian envelope, which is fine.
    let wave = 0.75 + 0.25 * (std::f32::consts::PI * (i as f32 - center)).cos();
    envelope * wave
}

/// Recording bars: react to audio level, gaussian taper across the row,
/// soft glow halo behind each bar at higher amplitudes. The audio level is
/// the dominant driver — only ~15 % of the bar height comes from the
/// per-bar phase animation, so silence reads as actually quiet and loud
/// speech reaches near the pill edge.
fn draw_bars(
    pixmap: &mut Pixmap,
    theme: &Theme,
    bar_color: [u8; 4],
    level: f32,
    anim: AnimState,
    pill_cy: f32,
    scale: f32,
) {
    let bar_w = BAR_W * scale;
    let bar_gap = BAR_GAP * scale;
    let bar_pitch = bar_w + bar_gap;
    let bar_block_w = BAR_COUNT as f32 * bar_w + (BAR_COUNT - 1) as f32 * bar_gap;
    let baseline = BAR_BASELINE * scale;
    let vpad = BAR_VPAD * scale;
    let surface_w = pixmap.width() as f32;
    // Track the *currently displayed* pill height so bars stay within the
    // pill while it's still growing during the spawn animation.
    let max_h = (anim.pill_height - vpad * 2.0).max(baseline + 2.0 * scale);
    let bar_x_start = (surface_w - bar_block_w) / 2.0;

    for i in 0..BAR_COUNT {
        let taper = taper_factor(i, BAR_COUNT);
        // Pure level-driven height. Each bar's center is anchored to
        // `pill_cy` and the bar grows symmetrically up and down. No
        // per-bar phase animation — that creates the illusion of bars
        // translating instead of expanding, which reads as "moving up
        // and down" rather than "amplitude".
        let effective = (level * taper).clamp(0.0, 1.0);
        let h = (baseline + effective * (max_h - baseline)).max(baseline);
        let bx = bar_x_start + i as f32 * bar_pitch;
        let by = pill_cy - h / 2.0;

        // Glow halo behind the bar — only visible above a small threshold.
        if effective > 0.02 {
            let glow_intensity = (effective * 0.9 + 0.1).clamp(0.0, 1.0);
            let glow_a = theme.glow[0] as f32 / 255.0 * glow_intensity * anim.bar_alpha;
            let glow_color = Color::from_rgba(
                theme.glow[1] as f32 / 255.0,
                theme.glow[2] as f32 / 255.0,
                theme.glow[3] as f32 / 255.0,
                glow_a.clamp(0.0, 1.0),
            )
            .unwrap_or(Color::TRANSPARENT);
            let glow_w = bar_w + 2.0 * scale;
            let glow_h = (h + 2.0 * scale).max(baseline + 2.0 * scale);
            if let Some(path) = build_stadium(bx - scale, pill_cy - glow_h / 2.0, glow_w, glow_h) {
                let mut paint = Paint {
                    anti_alias: true,
                    ..Default::default()
                };
                paint.set_color(glow_color);
                pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    Transform::identity(),
                    None,
                );
            }
        }

        if let Some(path) = build_stadium(bx, by, bar_w, h) {
            let mut paint = Paint {
                anti_alias: true,
                ..Default::default()
            };
            paint.set_color(theme_color(bar_color, anim.bar_alpha));
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }
}

/// Transcribing state: no audio level, just a center-out shimmer that
/// travels across the bar row to communicate "working on it" without
/// flat staticness.
fn draw_sweep(
    pixmap: &mut Pixmap,
    theme: &Theme,
    frame: u32,
    anim: AnimState,
    pill_cy: f32,
    scale: f32,
) {
    let bar_w = BAR_W * scale;
    let bar_gap = BAR_GAP * scale;
    let bar_pitch = bar_w + bar_gap;
    let bar_block_w = BAR_COUNT as f32 * bar_w + (BAR_COUNT - 1) as f32 * bar_gap;
    let baseline = BAR_BASELINE * scale;
    let vpad = BAR_VPAD * scale;
    let surface_w = pixmap.width() as f32;
    let max_h = (anim.pill_height - vpad * 2.0).max(baseline + 2.0 * scale);
    let bar_x_start = (surface_w - bar_block_w) / 2.0;

    // Sliding focus point that pings back and forth across the row.
    let cycle = (BAR_COUNT as i32) * 2 - 2;
    let pos = ((frame / 3) as i32) % cycle.max(1);
    let active = if pos < BAR_COUNT as i32 {
        pos as f32
    } else {
        (cycle - pos) as f32
    };

    for i in 0..BAR_COUNT {
        let taper = taper_factor(i, BAR_COUNT);
        let dist = (i as f32 - active).abs();
        // Bell-shaped intensity centered on `active`, ~3 bars wide.
        let intensity = (-dist * dist / 4.0).exp().max(0.15);
        let dynamic = intensity * taper;
        let h = (baseline + dynamic * (max_h - baseline) * 0.85).max(baseline);
        let bx = bar_x_start + i as f32 * bar_pitch;
        let by = pill_cy - h / 2.0;

        let bar_a = theme.trans_bar[0] as f32 / 255.0 * (0.3 + 0.7 * intensity) * anim.bar_alpha;
        let bar_color = Color::from_rgba(
            theme.trans_bar[1] as f32 / 255.0,
            theme.trans_bar[2] as f32 / 255.0,
            theme.trans_bar[3] as f32 / 255.0,
            bar_a.clamp(0.0, 1.0),
        )
        .unwrap_or(Color::TRANSPARENT);

        if let Some(path) = build_stadium(bx, by, bar_w, h) {
            let mut paint = Paint {
                anti_alias: true,
                ..Default::default()
            };
            paint.set_color(bar_color);
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 100;
    const H: u32 = 64;

    #[test]
    fn parses_hyprland_focused_output() {
        let output = parse_hyprland_active_workspace_output(
            br#"{"id":2,"name":"2","monitor":"eDP-1","monitorID":0}"#,
        );
        assert_eq!(output.as_deref(), Some("eDP-1"));
    }

    #[test]
    fn rejects_hyprland_workspace_without_monitor() {
        assert_eq!(parse_hyprland_active_workspace_output(br#"{"id":2}"#), None);
        assert_eq!(
            parse_hyprland_active_workspace_output(br#"{"monitor":""}"#),
            None
        );
    }

    fn fresh_pixmap() -> Pixmap {
        Pixmap::new(W, H).unwrap()
    }

    fn shown() -> AnimState {
        AnimState {
            pill_height: H as f32,
            pill_alpha: 1.0,
            bar_alpha: 1.0,
            bars_locked: false,
        }
    }
    fn hidden() -> AnimState {
        AnimState {
            pill_height: SPAWN_PILL_MIN_H,
            pill_alpha: 0.0,
            bar_alpha: 0.0,
            bars_locked: true,
        }
    }

    #[test]
    fn nonblocking_wayland_reads_only_retry_would_block() {
        use wayland_client::backend::WaylandError;
        assert!(tolerate_would_block::<()>(Err(WaylandError::Io(
            std::io::ErrorKind::WouldBlock.into()
        )))
        .is_ok());
        assert!(tolerate_would_block::<()>(Err(WaylandError::Io(
            std::io::ErrorKind::BrokenPipe.into()
        )))
        .is_err());
    }

    #[test]
    fn completion_clears_visible_overlay_on_next_frame() {
        let (state_tx, state_rx) = mpsc::channel();
        let (_level_tx, level_rx) = mpsc::channel();
        let mut renderer =
            OverlayRenderer::new(state_rx, level_rx, W, H, Theme::ember(), 1.0).unwrap();
        state_tx.send(State::Recording).unwrap();
        renderer.apply_state_updates();
        renderer.spawn_started = Instant::now() - Duration::from_secs(1);
        renderer.draw_frame();
        assert!(renderer.pixmap.data().chunks_exact(4).any(|px| px[3] != 0));
        state_tx.send(State::Transcribing).unwrap();
        renderer.draw_frame();
        state_tx.send(State::Idle).unwrap();
        renderer.draw_frame();
        assert!(renderer.pixmap.data().iter().all(|byte| *byte == 0));
        state_tx.send(State::Recording).unwrap();
        renderer.apply_state_updates();
        assert_eq!(renderer.visible_state, State::Recording);
        assert!(renderer.spawn_in);
    }

    #[test]
    fn idle_draw_is_transparent() {
        let mut pm = fresh_pixmap();
        let t = Theme::ember();
        draw_overlay(&mut pm, State::Idle, 0, 0.0, hidden(), &t, 1.0);
        assert!(pm.data().iter().all(|b| *b == 0));
    }

    #[test]
    fn faded_out_draw_is_transparent() {
        let mut pm = fresh_pixmap();
        let t = Theme::ember();
        draw_overlay(&mut pm, State::Recording, 0, 1.0, hidden(), &t, 1.0);
        assert!(pm.data().iter().all(|b| *b == 0));
    }

    #[test]
    fn active_draw_has_visible_pixels() {
        let mut pm = fresh_pixmap();
        let t = Theme::ember();
        draw_overlay(&mut pm, State::Recording, 0, 1.0, shown(), &t, 1.0);
        // tiny-skia stores premultiplied RGBA; alpha lives in the 4th byte.
        assert!(pm.data().chunks_exact(4).any(|px| px[3] != 0));
    }

    #[test]
    fn speaking_uses_distinct_bar_color_from_recording() {
        // Recording bars are amber (#F97316: high R, mid G, low B); speaking
        // bars are emerald (#34D399: low R, high G, mid B). Confirm the
        // speaking frame paints green-dominant bar pixels that the recording
        // frame does not.
        fn green_dominant(data: &[u8]) -> usize {
            data.chunks_exact(4)
                .filter(|px| px[1] > 150 && px[0] < 120 && px[1] > px[0])
                .count()
        }

        let t = Theme::ember();
        let mut rec = fresh_pixmap();
        let mut spk = fresh_pixmap();
        draw_overlay(&mut rec, State::Recording, 0, 1.0, shown(), &t, 1.0);
        draw_overlay(&mut spk, State::Speaking, 0, 1.0, shown(), &t, 1.0);
        let rec_green = green_dominant(rec.data());
        let spk_green = green_dominant(spk.data());
        assert!(
            spk_green > rec_green,
            "speaking should paint more green-dominant pixels than recording \
             (recording={rec_green}, speaking={spk_green})"
        );
    }

    #[test]
    fn synthesizing_draws_visible_pixels() {
        let mut pm = fresh_pixmap();
        let t = Theme::ember();
        draw_overlay(&mut pm, State::Synthesizing, 0, 0.0, shown(), &t, 1.0);
        assert!(pm.data().chunks_exact(4).any(|px| px[3] != 0));
    }

    #[test]
    fn taper_is_strongest_in_center() {
        let center = taper_factor(BAR_COUNT / 2, BAR_COUNT);
        let edge_left = taper_factor(0, BAR_COUNT);
        let edge_right = taper_factor(BAR_COUNT - 1, BAR_COUNT);
        assert!(center > edge_left);
        assert!(center > edge_right);
        assert!(edge_left < 0.5);
        assert!(edge_right < 0.5);
    }

    #[test]
    fn ease_curves_hit_endpoints() {
        assert!((ease_in_cubic(0.0) - 0.0).abs() < 1e-6);
        assert!((ease_in_cubic(1.0) - 1.0).abs() < 1e-6);
        assert!((ease_out_back(0.0, 0.4) - 0.0).abs() < 1e-6);
        assert!((ease_out_back(1.0, 0.4) - 1.0).abs() < 1e-6);
        // The "back" curve is supposed to peak above 1 in the middle.
        assert!(ease_out_back(0.85, 0.4) > 1.0);
    }

    #[test]
    fn x11_position_in_rect_uses_monitor_bounds() {
        assert_eq!(
            position_in_rect(1920, 360, 1920, 1200, 100, 40),
            (2830, 1504)
        );
        assert_eq!(position_in_rect(3840, 0, 1200, 1920, 100, 40), (4390, 1864));
    }

    #[test]
    fn x11_point_in_rect_uses_half_open_bounds() {
        assert!(point_in_rect(2830, 1504, 1920, 360, 1920, 1200));
        assert!(!point_in_rect(3840, 1504, 1920, 360, 1920, 1200));
    }

    #[test]
    fn silence_draws_minimal_baseline() {
        // Recording bars in the ember theme are amber (#F97316). tiny-skia
        // pixmap pixels are premultiplied RGBA in memory order [R, G, B, A];
        // count amber-dominant pixels (high R, mid G, low B) to measure
        // bar area independent of the bg pill.
        fn amber_pixels(data: &[u8]) -> usize {
            data.chunks_exact(4)
                .filter(|px| px[0] > 200 && px[1] > 70 && px[1] < 180 && px[2] < 60)
                .count()
        }

        let t = Theme::ember();
        let mut quiet = fresh_pixmap();
        let mut loud = fresh_pixmap();
        draw_overlay(&mut quiet, State::Recording, 0, 0.0, shown(), &t, 1.0);
        draw_overlay(&mut loud, State::Recording, 0, 1.0, shown(), &t, 1.0);
        let count_quiet = amber_pixels(quiet.data());
        let count_loud = amber_pixels(loud.data());
        assert!(
            count_loud > count_quiet,
            "loud audio should fill more bar area than silence (silence={count_quiet}, loud={count_loud})"
        );
    }
}
