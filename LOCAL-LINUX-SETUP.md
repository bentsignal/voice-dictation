# Local voice dictation on NixOS and other Linux distributions

This fork supports the same private, on-device dictation workflow on both of
Shawn's KDE Wayland machines. The application code is portable Rust; only the
host integration differs:

- NixOS uses [`nixos/voice-dictation.nix`](nixos/voice-dictation.nix).
- Arch, EndeavourOS, and other systemd distributions use
  [`scripts/setup-local-parakeet.sh`](scripts/setup-local-parakeet.sh).

Both paths provide Right Alt press-to-toggle dictation, the bottom-centre HUD,
Unicode text injection, and local Parakeet TDT 0.6B v3 INT8 transcription via
sherpa-onnx. Audio never leaves the machine.

## Arch / EndeavourOS

Install the host build tools and `uv` once:

```console
sudo pacman -S --needed base-devel rust alsa-lib libxkbcommon clang uv
```

Then run the rootless per-user installer from this checkout:

```console
./scripts/setup-local-parakeet.sh
```

The script builds the sidecar-only whisrs variant, creates an isolated Python
3.12 environment, downloads and verifies the official sherpa-onnx INT8 model,
and enables the two systemd user services. Existing whisrs configuration is
never overwritten. When available, it selects the Pulse input explicitly so
PipeWire's configured default source is honored instead of a hardware ALSA
fallback. The included user service selects the US XKB layout used on these two
machines; change `XKB_DEFAULT_LAYOUT` in `whisrs-local.service` when installing
on a workstation with a different layout.

The built-in Right Alt hotkey reads Linux input events. If the account is not
already in the `input` group, run `sudo usermod -aG input "$USER"` and log out
and back in once. `/dev/uinput` also needs the rule from
[`contrib/99-whisrs.rules`](contrib/99-whisrs.rules); desktop logind ACLs often
already provide access.

Useful checks:

```console
systemctl --user status parakeet-sidecar whisrs
curl http://127.0.0.1:8765/health
~/.local/bin/whisrs status
```

## NixOS

The existing NixOS integration and its machine-specific notes remain in
[`LOCAL-NIXOS-SETUP.md`](LOCAL-NIXOS-SETUP.md). It builds the same source tree
through `flake-package.nix` and manages equivalent user services declaratively.

## Background batches for long dictation

Set `chunk_seconds = 30` in `[asr-sidecar]` in `~/.config/whisrs/config.toml`,
then restart `whisrs.service` while idle. Dictation decodes in the background
with requests of at most 30 seconds, choosing the quietest 100 ms in the last
five seconds for each boundary. Short recordings still use one request. Text
is joined in order and inserted once on stop using the existing output mode.
Set the value to `0` and restart to restore whole-recording transcription.

Requests run sequentially and time out after 120 seconds. On a request failure,
no partial transcript is inserted; capture continues and the complete audio
is saved by the existing recovery workflow when you stop. Cancellation drops
the background work and inserts nothing. Audio stays local with the local URL.
Raw PCM is still retained in memory for recovery (about 1.92 MB/minute, plus
allocation overhead), but model inference receives only a bounded chunk.

Chunk boundaries can change punctuation or recognition, especially without
pauses. To compare the local recognizer with a 16 kHz mono PCM16 WAV:

```console
python3 scripts/benchmark-chunks.py sample.wav --seconds 90
```

The script reports durations and agreement between word sequences, not
accuracy against a human transcript. `--repeat` fills the duration by repeating
a short fixture. It sends requests only to `127.0.0.1:8765` and prints no text
from the recording. Its final-chunk timing estimates remaining inference work
when earlier chunks have finished; it excludes capture shutdown and insertion.
