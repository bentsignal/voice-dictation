#!/usr/bin/env python3
"""Compare local whole-recording and chunked inference without printing speech."""

import argparse
import array
import difflib
import io
import json
import re
import time
import urllib.request
import wave

RATE = 16000


def transcribe(samples):
    wav = io.BytesIO()
    with wave.open(wav, "wb") as writer:
        writer.setparams((1, 2, RATE, 0, "NONE", "none"))
        writer.writeframes(samples.tobytes())
    boundary = "whisrs-benchmark-boundary"
    header = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    )
    data = header.encode() + wav.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8765/transcribe",
        data,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        text = json.load(response)["text"]
    return text, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="16 kHz mono PCM16 WAV")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--repeat", action="store_true", help="repeat a short fixture")
    args = parser.parse_args()
    if args.seconds <= 0 or args.start < 0:
        parser.error("seconds must be positive and start must be nonnegative")
    with wave.open(args.wav, "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (RATE, 1, 2):
            parser.error("input must be 16 kHz mono PCM16")
        reader.setpos(int(args.start * RATE))
        raw = reader.readframes(RATE * args.seconds)
    if not raw:
        parser.error("input selection is empty")
    if args.repeat:
        target = args.seconds * RATE * 2
        raw = (raw * (target // len(raw) + 1))[:target]
    samples = array.array("h", raw)
    whole, elapsed = transcribe(samples)
    print(json.dumps({
        "audio_seconds": len(samples) / RATE,
        "whole_request_seconds": round(elapsed, 3),
        "words": len(whole.split()),
    }), flush=True)

    parts, times, durations = [], [], []
    while samples:
        end = len(samples)
        if end >= 30 * RATE:
            # Same quiet-frame boundary selection as chunked_batch.rs.
            frame = RATE // 10
            end = min(
                range(25 * RATE, 30 * RATE, frame),
                key=lambda n: sum(x * x for x in samples[n:n + frame]),
            ) + frame // 2
        text, elapsed = transcribe(samples[:end])
        parts.append(text)
        times.append(elapsed)
        durations.append(end / RATE)
        samples = samples[end:]
    words = lambda text: re.findall(r"\w+", text.lower())
    agreement = difflib.SequenceMatcher(
        None, words(whole), words(" ".join(parts)), autojunk=False
    ).ratio()
    print(json.dumps({
        "chunk_audio_seconds": durations,
        "chunk_request_seconds": [round(t, 3) for t in times],
        "total_chunk_compute_seconds": round(sum(times), 3),
        "final_chunk_seconds": round(times[-1], 3),
        "word_sequence_similarity": round(agreement, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
