"""Clips and expected frames for web/src/decode.browser.test.ts.

Builds tiny clips whose frames say which index they are, then records what
timeslice.probe and timeslice.load_video make of each under a range of options:
the size, which source frames were kept, and the frames themselves.

    uv run python web/test/fixtures/decode.py

Writes the clips, decode.json and frames.bin.gz to decode/ beside this script.
decode.json lists, per case, the source index of every frame load_video returned
and where its pixels (rgb24, rows top first) start in the decompressed
frames.bin; the errors Python raises for files it cannot use; files Python
reads and a browser cannot (a codec, a container); and what clip_seconds gives
for a range of arguments.
"""

import gzip
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import timeslice  # noqa: E402

OUT = Path(__file__).resolve().parent / "decode"
WIDTH, HEIGHT, FRAMES = 64, 48, 40

# Each frame is one colour from a palette of well-separated colours, so its
# index survives lossy coding and resizing, made lighter or darker by a
# different amount in each quadrant, so its orientation shows too. Adding the
# same to red, green and blue changes only luma, so the frame has no chroma
# edges: at those, ffmpeg's rgb24 (nearest chroma) and a browser (interpolated
# chroma) legitimately differ, by some 20 levels.
LEVELS = np.array([44, 104, 164, 224])
PALETTE = np.array([[r, g, b] for r in LEVELS for g in LEVELS for b in LEVELS])
PALETTE = PALETTE[np.random.default_rng(0).permutation(len(PALETTE))[:FRAMES]]
QUADRANT_SHADES = (-30, -10, 10, 30)


def source_frames():
    frames = np.empty((FRAMES, HEIGHT, WIDTH, 3), np.uint8)
    h, w = HEIGHT // 2, WIDTH // 2
    for i in range(FRAMES):
        for q, shade in enumerate(QUADRANT_SHADES):
            y, x = divmod(q, 2)
            frames[i, y * h:(y + 1) * h, x * w:(x + 1) * w] = PALETTE[i] + shade
    return frames


# H.264 and VP9 at 4:2:0, tagged BT.709 so ffmpeg and the browser read the
# same colour matrix. One thread and bitexact keep the files reproducible.
COLOUR = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
          "-color_range", "tv", "-threads", "1"]
BITEXACT = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact"]
# Fixed B-frames, so decode order differs from presentation order.
H264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "12",
        "-bf", "3", "-x264-params", "b-adapt=0"]
VP9 = ["-c:v", "libvpx-vp9", "-crf", "12", "-b:v", "0", "-g", "12",
       "-deadline", "good", "-cpu-used", "4", "-row-mt", "0"]

# Variable rates: timestamps rewritten by setpts and passed through unchanged.
PASSTHROUGH = ["-fps_mode", "passthrough"]
# Frames 10-19 last twice as long.
SLOW_MIDDLE = ["-vf", "setpts='(N+clip(N-10,0,10))/(30*TB)'", *PASSTHROUGH,
               "-video_track_timescale", "15360"]
# About 29.93 fps, each frame up to 0.3 ms early or late, at 90 kHz.
JITTER = ["-vf", "settb=1/90000,setpts='N*3007+trunc(25*sin(N*2.3))'", *PASSTHROUGH,
          "-enc_time_base", "1/90000", "-video_track_timescale", "90000"]
JITTER_MS = ["-vf", "settb=1/1000,setpts='N*33+trunc(2*sin(N*1.9))'", *PASSTHROUGH,
             "-enc_time_base", "1/1000"]
# A timecode track, which ffmpeg counts as a stream and Mediabunny doesn't list.
TIMECODE = ["-timecode", "00:00:00:00", "-write_tmcd", "1"]
# No edit list, so the frame shown first starts at its composition offset.
NO_EDIT_LIST = ["-use_editlist", "0"]
# B-frames as x264 lays them out by default, declaring a reorder delay of 2 and using 1.
H264_BF2 = ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "12",
            "-bf", "2", "-x264-params", "b-adapt=0"]
# At a time base of 1/30, frames 9 and 24 last 3 and 4 ticks.
GAPS = ["-vf", "settb=1/30,setpts='N+2*gte(N,10)+3*gte(N,25)'", *PASSTHROUGH,
        "-enc_time_base", "1/30", "-video_track_timescale", "30"]
# Twelve frames of 517 ticks at 15360 Hz, the last four lasting twice as long: too few for
# ffmpeg's common-step rate, so only the sample table says whether it is constant.
LONG_TAIL = ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "12", "-bf", "0",
             "-frames:v", "12",
             "-vf", "settb=1/15360,setpts='N*517+gte(N,9)*(N-8)*517'", *PASSTHROUGH,
             "-enc_time_base", "1/15360", "-video_track_timescale", "15360"]


# About 30 fps at a time base of 1/1000, frame 11 coming 100 ms after frame 10, as a
# browser recording stalls.
STALL = ["-vf", "settb=1/1000,setpts='trunc(N*100/3+0.5)+67*gt(N,10)'", *PASSTHROUGH,
         "-enc_time_base", "1/1000"]
# The frames twice over at a time base of 1/60, 30 fps for 45 frames and 20 after: at a
# time base this coarse ffmpeg judges the rate from 40 steps, not the header's 99 samples.
TS60 = ["-vf", f"loop=loop=1:size={FRAMES},settb=1/60,setpts='if(lt(N,45),2*N,90+3*(N-45))'",
        *PASSTHROUGH, "-enc_time_base", "1/60", "-video_track_timescale", "60"]


def vfr600(late):
    """The frames three times over at a time base of 1/600, as phones and screen recorders
    write, the step to frame 30 lasting 25 ticks and to frame `late` 27: ffmpeg judges
    the rate from the first 99 frames, and paces to the average rate where it judges
    one over 210 fps."""
    return ["-vf", f"loop=loop=2:size={FRAMES},settb=1/600,setpts='N*20+5*gte(N,30)+7*gte(N,{late})'",
            *PASSTHROUGH, "-enc_time_base", "1/600", "-video_track_timescale", "600"]


# The frames three times over at 240 fps and a time base of 1/2400, frame 30 a frame late:
# ffmpeg judges 240 fps and, the average staying over 70, paces to that.
VFR2400 = ["-vf", f"loop=loop=2:size={FRAMES},settb=1/2400,setpts='N*10+10*gte(N,30)'",
           *PASSTHROUGH, "-enc_time_base", "1/2400", "-video_track_timescale", "2400"]

# About 30 fps at a time base of 1/1000, each frame up to 8 ms early or late: too uneven for
# any standard rate, so ffmpeg takes the rate the codec declares, if the time base can count
# it, or else the time base, whose 1000 fps sends it to the average rate.
UNEVEN_MS = ["-vf", "settb=1/1000,setpts='N*33+trunc(8*sin(N*2.3))'", *PASSTHROUGH,
             "-enc_time_base", "1/1000", "-video_track_timescale", "1000"]
# The same at a time base of 1/600, as phones write.
UNEVEN_600 = ["-vf", "settb=1/600,setpts='N*20+trunc(5*sin(N*2.3))'", *PASSTHROUGH,
              "-enc_time_base", "1/600", "-video_track_timescale", "600"]
FRAGMENTED = ["-movflags", "+frag_keyframe+empty_moov"]
# Without B-frames, as browsers record.
H264_NO_B = ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "12", "-bf", "0"]
# x264 declares the time base's rate for timestamps passed through, unless told the frames
# come at a constant 30 fps.
H264_DECLARED_30 = ["-c:v", "libx264", "-preset", "medium", "-crf", "12", "-g", "12", "-bf", "3",
                    "-x264-params", "b-adapt=0:force-cfr=1:fps=30"]
# x265 declares the input's 30 fps; Apple's encoder, like x265 told not to, declares none.
X265 = ["-c:v", "libx265", "-preset", "medium", "-crf", "12", "-g", "12", "-bf", "3", "-tag:v", "hvc1"]
HEVC = [*X265, "-x265-params", "log-level=error"]
HEVC_UNTIMED = [*X265, "-x265-params", "log-level=error:vui-timing-info=0"]
# The frames three times over at 240 fps and a time base of 1/1000, the stream stating 30.
AT_240 = ["-vf", f"loop=loop=2:size={FRAMES},settb=1/1000,setpts='trunc(N*25/6+0.5)'", *PASSTHROUGH,
          "-enc_time_base", "1/1000"]
# ffmpeg's mov demuxer drops VP9 packets' durations and makes them up from r_frame_rate, here the
# time base's 1000 fps, but for those it reads while probing for that rate, unless it seeks and
# reads them again. UNEVEN_MS twice over, frame 41 coming 133 ms late: probing reads 41 frames.
UNEVEN_STALL = ["-vf", f"loop=loop=1:size={FRAMES},settb=1/1000,"
                "setpts='N*33+trunc(8*sin(N*2.3))+133*gte(N,41)'", *PASSTHROUGH,
                "-enc_time_base", "1/1000", "-video_track_timescale", "1000"]
# About 5 fps at a time base of 1/1000, frame 23 coming 800 ms late: probing stops at 5 s of
# frames at the average rate, after 24.
SLOW_STALL = ["-vf", "settb=1/1000,setpts='N*200+trunc(40*sin(N*2.3))+800*gte(N,23)'", *PASSTHROUGH,
              "-enc_time_base", "1/1000", "-video_track_timescale", "1000"]
# The frames three times over at 30 fps and a time base of 1/600, frame 20 two frames late and
# frames from 45 on up to 8 ticks early or late: too uneven for the header's 99 samples to give
# ffmpeg a rate, which it finds in the 41 packets it then probes, each lasting a tick.
COARSE_STALL = ["-vf", f"loop=loop=2:size={FRAMES},settb=1/600,"
                "setpts='N*20+40*gte(N,20)+gte(N,45)*trunc(8*sin(N*2.3))'", *PASSTHROUGH,
                "-enc_time_base", "1/600", "-video_track_timescale", "600"]
# 5 fps for 32 frames at a time base of 1/1000, then 10 fps.
FIVE_THEN_TEN = ["-vf", "settb=1/1000,setpts='N*200-clip(N-32,0,8)*100'", *PASSTHROUGH,
                 "-enc_time_base", "1/1000"]
# The frames twice over at 10 fps and a time base of 1/1000, frame 60 coming 300 ms late.
TEN_LATE_60 = ["-vf", f"loop=loop=1:size={FRAMES},settb=1/1000,setpts='N*100+300*gte(N,60)'", *PASSTHROUGH,
               "-enc_time_base", "1/1000"]


def ffmpeg(*args, input=None):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], input=input, check=True)


# Steps that remux a clip from one file into another.
def rotate(degrees, *keep):
    """Set the display matrix on a stream copy, so the coded frames stay
    landscape and only the metadata says to turn them."""
    return lambda src, dst: ffmpeg("-display_rotation:v:0", degrees, "-i", src, "-c", "copy",
                                   *BITEXACT, *keep, dst)


def add_audio(codec, video_offset=None):
    """Add a second stream, which changes how ffmpeg paces the frames it
    outputs. AAC's priming samples put the audio's first timestamp just before
    0; Opus in Matroska records its delay instead. With video_offset the video
    starts that many seconds after the audio."""
    offset = [] if video_offset is None else ["-itsoffset", video_offset]
    sine = "sine=frequency=440:duration=1.7:sample_rate=44100"
    return lambda src, dst: ffmpeg(*offset, "-i", src, "-f", "lavfi", "-i", sine, "-map", "0:v",
                                   "-map", "1:a", "-c:v", "copy", "-c:a", codec, "-b:a", "16k",
                                   *BITEXACT, dst)


def add_subtitles(video_offset=None):
    """Add a WebVTT stream from 0, which ffmpeg counts and Mediabunny doesn't list,
    the video starting video_offset seconds later."""
    offset = [] if video_offset is None else ["-itsoffset", video_offset]

    def step(src, dst):
        cues = dst.with_suffix(".vtt")
        cues.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n")
        ffmpeg(*offset, "-i", src, "-i", cues, "-map", "0:v", "-map", "1:s", "-c:v", "copy",
               "-c:s", "webvtt", *BITEXACT, dst)
        cues.unlink()
    return step


def add_timecode(video_offset):
    """Add a timecode track from 0, the video starting video_offset seconds later."""
    return lambda src, dst: ffmpeg("-itsoffset", video_offset, "-i", src, "-c", "copy", *TIMECODE,
                                   *BITEXACT, dst)


def lengthen_last(src, dst):
    """Make the last sample last 1034 ticks too, as its packet's duration says."""
    ffmpeg("-i", src, "-c", "copy", "-bsf:v", "setts=duration='if(eq(N,11),1034,DURATION)'",
           *BITEXACT, dst)


def cut(start, duration):
    """Cut by stream copy, keeping the frames back to the key frame before `start` and
    an edit list from `start` (within a frame) for `duration`."""
    return lambda src, dst: ffmpeg("-ss", start, "-i", src, "-t", duration, "-c", "copy",
                                   *BITEXACT, dst)


def without_default_durations(src, dst):
    """Turn the tracks' DefaultDuration elements into Void, as browser recorders write
    none, so that no block has a duration."""
    data = bytearray(Path(src).read_bytes())
    clusters = data.index(bytes([0x1f, 0x43, 0xb6, 0x75]))
    at = data.find(bytes([0x23, 0xe3, 0x83]), 0, clusters)
    assert at >= 0
    while at >= 0:
        size = data[at + 3]
        assert size & 0x80, "a one-byte size"
        data[at:at + 2] = bytes([0xec, 0x80 | (size & 0x7f) + 2])
        at = data.find(bytes([0x23, 0xe3, 0x83]), at, clusters)
    Path(dst).write_bytes(data)


def default_duration(ns):
    """Set the first track's DefaultDuration to `ns` nanoseconds, in the bytes it already has."""
    def step(src, dst):
        data = bytearray(Path(src).read_bytes())
        clusters = data.index(bytes([0x1f, 0x43, 0xb6, 0x75]))
        at = data.find(bytes([0x23, 0xe3, 0x83]), 0, clusters)
        assert at >= 0 and data[at + 3] & 0x80, "a DefaultDuration with a one-byte size"
        size = data[at + 3] & 0x7f
        data[at + 4:at + 4 + size] = ns.to_bytes(size, "big")
        Path(dst).write_bytes(data)
    return step


def add_empty_audio(src, dst):
    """Add an audio stream with no packets, which ffmpeg gives no start time.
    A fragmented MP4 keeps it, where a plain one would leave it out."""
    ffmpeg("-i", src, "-f", "lavfi", "-t", 1, "-i", "anullsrc=r=44100", "-map", "0:v",
           "-map", "1:a", "-c:v", "copy", "-af", "aselect=0", "-c:a", "aac",
           "-movflags", "+frag_keyframe+empty_moov", *BITEXACT, dst)


# Name: (input rate, encoder and options, pixel format, then the steps that remux it).
CLIPS = {
    "h264-30.mp4": (Fraction(30), H264, "yuv420p"),
    "h264-ntsc.mp4": (Fraction(30000, 1001), H264, "yuv420p"),
    "vp9-25.webm": (Fraction(25), VP9, "yuv420p"),
    "h264-rot90.mp4": (Fraction(30), H264, "yuv420p", rotate(90)),
    "h264-30-aac.mp4": (Fraction(30), H264, "yuv420p", add_audio("aac")),
    "h264-60-aac.mp4": (Fraction(60), H264, "yuv420p", add_audio("aac")),
    "h264-vfr.mp4": (Fraction(30), H264 + SLOW_MIDDLE, "yuv420p"),
    "h264-vfr-aac.mp4": (Fraction(30), H264 + SLOW_MIDDLE, "yuv420p", add_audio("aac")),
    "h264-jitter.mp4": (Fraction(30), H264 + JITTER, "yuv420p"),
    "vp9-jitter.webm": (Fraction(30), VP9 + JITTER_MS, "yuv420p"),
    "h264-rot90-aac.mp4": (Fraction(30), H264, "yuv420p", rotate(90), add_audio("aac")),
    "h264-tmcd.mp4": (Fraction(30), H264 + TIMECODE, "yuv420p"),
    "vp9-subs.webm": (Fraction(25), VP9, "yuv420p", add_subtitles()),
    "vp9-opus.webm": (Fraction(25), VP9, "yuv420p", add_audio("libopus")),
    "h264-noedit.mp4": (Fraction(30), H264 + NO_EDIT_LIST, "yuv420p"),
    "h264-vfr-late.mp4": (Fraction(30), H264 + SLOW_MIDDLE, "yuv420p",
                          add_audio("aac", video_offset=0.5)),
    "h264-empty-audio.mp4": (Fraction(30), H264, "yuv420p", add_empty_audio),
    "vp9-alpha.webm": (Fraction(25), VP9, "yuva420p"),
    # The alpha flag is stream metadata, which bitexact drops on a copy.
    "vp9-alpha-rot90.webm": (Fraction(25), VP9, "yuva420p",
                             rotate(90, "-metadata:s:v:0", "alpha_mode=1")),
    # ffmpeg starts the file at a subtitle or data stream less than a second before the video.
    "h264-tmcd-late.mp4": (Fraction(30), H264, "yuv420p", add_timecode(0.5)),
    "h264-tmcd-late15.mp4": (Fraction(30), H264, "yuv420p", add_timecode(1.5)),
    "vp9-subs-late.webm": (Fraction(25), VP9, "yuv420p", add_subtitles(video_offset=0.4)),
    "h264-bf2-vfr.mp4": (Fraction(30), H264_BF2 + GAPS, "yuv420p"),
    "h264-long-tail.mp4": (Fraction(30), LONG_TAIL, "yuv420p", lengthen_last),
    "h264-vfr600.mp4": (Fraction(30), H264 + vfr600(55), "yuv420p"),
    "h264-vfr600-240.mp4": (Fraction(30), H264 + vfr600(40), "yuv420p"),
    "h264-vfr2400.mp4": (Fraction(30), H264 + VFR2400, "yuv420p"),
    "h264-cut-aac.mp4": (Fraction(30), H264, "yuv420p", add_audio("aac"), cut(0.51, 0.5)),
    "vp9-opus-stall.webm": (Fraction(30), VP9 + STALL, "yuv420p", add_audio("libopus"),
                            without_default_durations),
    "h264-ts60.mp4": (Fraction(30), H264 + TS60, "yuv420p"),
    # The 1000 fps declared is too fine for the time base, so ffmpeg takes the time base, and
    # the average.
    "h264-uneven.mp4": (Fraction(30), H264 + UNEVEN_MS, "yuv420p"),
    # The same, the average over the fragments.
    "h264-uneven-frag.mp4": (Fraction(30), H264_NO_B + UNEVEN_MS + FRAGMENTED, "yuv420p"),
    # ffmpeg takes the 30 fps declared, doubled as H.264 counts fields.
    "h264-uneven-30.mp4": (Fraction(30), H264_DECLARED_30 + UNEVEN_MS, "yuv420p"),
    # ffmpeg takes the 30 fps declared.
    "hevc-uneven-30.mp4": (Fraction(30), HEVC + UNEVEN_MS, "yuv420p"),
    # ffmpeg takes the time base, 600 fps, and paces to the average.
    "hevc-uneven-untimed.mp4": (Fraction(30), HEVC_UNTIMED + UNEVEN_600, "yuv420p"),
    # ffmpeg judges 240 fps from the timestamps and paces to the 4 fps the default duration gives.
    "vp9-240-at-4.webm": (Fraction(30), VP9 + AT_240, "yuv420p", default_duration(250_000_000)),
    # Probing reads every packet, so none lasts 1 ms unless ffmpeg seeks.
    "vp9-uneven-frag.mp4": (Fraction(30), VP9 + UNEVEN_MS + FRAGMENTED, "yuv420p"),
    # The late frame is the first to last 1 ms.
    "vp9-uneven-stall.mp4": (Fraction(30), VP9 + UNEVEN_STALL, "yuv420p"),
    # The late frame is the last read while probing.
    "vp9-slow-stall.mp4": (Fraction(5), VP9 + SLOW_STALL, "yuv420p"),
    # The late frame lasts a tick.
    "vp9-600-stall.mp4": (Fraction(30), VP9 + COARSE_STALL, "yuv420p"),
    # Reading the header, ffmpeg judges 10 fps, from which it makes up every packet's duration.
    "vp9-5-10.mp4": (Fraction(10), VP9 + FIVE_THEN_TEN + ["-video_track_timescale", "1000"], "yuv420p"),
    # Without durations, each declaring 30 fps: ffmpeg makes up every H.264 packet's duration
    # from that, as it parses the stream, but only the HEVC packets it reads while probing.
    "h264-30-at-10.mkv": (Fraction(30), H264_DECLARED_30 + TEN_LATE_60, "yuv420p", without_default_durations),
    "hevc-30-at-10.mkv": (Fraction(30), HEVC + TEN_LATE_60, "yuv420p", without_default_durations),
    # Without packet durations or an average rate, ffmpeg stops probing at 5 s of decode
    # timestamps, so judges 5 fps.
    "vp9-5-10.webm": (Fraction(10), VP9 + FIVE_THEN_TEN, "yuv420p", without_default_durations),
    "hevc-untimed-5-10.mkv": (Fraction(10), HEVC_UNTIMED + FIVE_THEN_TEN, "yuv420p", without_default_durations),
    # x265 declares 10 fps, from which ffmpeg makes up durations, so it probes all the packets.
    "hevc-5-10.mkv": (Fraction(10), HEVC + FIVE_THEN_TEN, "yuv420p", without_default_durations),
}


def with_alpha(frames):
    """The frames with an alpha channel, which ffmpeg's rgb24 ignores: clear in
    the top left quadrant and half clear in the bottom right."""
    alpha = np.full(frames.shape[:3] + (1,), 255, np.uint8)
    alpha[:, :HEIGHT // 2, :WIDTH // 2] = 0
    alpha[:, HEIGHT // 2:, WIDTH // 2:] = 128
    return np.concatenate([frames, alpha], axis=3)


def encode(name, frames, rate, codec, pix_fmt, *steps):
    path = OUT / name
    rgba = pix_fmt.startswith("yuva")
    src = path.with_suffix(".0" + path.suffix) if steps else path
    ffmpeg("-f", "rawvideo", "-pix_fmt", "rgba" if rgba else "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
           "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-", *codec, *COLOUR,
           "-pix_fmt", pix_fmt, *BITEXACT, src,
           input=(with_alpha(frames) if rgba else frames).tobytes())
    for i, step in enumerate(steps, 1):
        dst = path if i == len(steps) else path.with_suffix(f".{i}" + path.suffix)
        step(src, dst)
        src.unlink()
        src = dst
    return path


def time_base(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=time_base", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    # A timecode track makes a stream group, which ffprobe lists the stream under too.
    return Fraction(out.stdout.split()[0].strip(","))


def option_sets(name, rate, tick):
    """load_video options to try on a clip whose frames last 1/rate seconds and
    whose timestamps count in ticks of `tick` seconds: full and preview quality,
    scaled, half and third rate, and windows that start and end on, between and
    within a tick of frames. The later clips each try a few options that bear
    on what sets them apart."""
    frame = 1 / rate
    if name in ("h264-30-aac.mp4", "h264-60-aac.mp4"):
        return [{}, {"time_scale": 0.5}, {"start": 0.51}, {"start": 0.51, "time_scale": 0.5},
                {"start": 0.51, "duration": 0.15}, {"start": 0.51, "duration": 0.2, "time_scale": 2}]
    if name in ("h264-vfr.mp4", "h264-vfr-aac.mp4"):
        return [{}, {"time_scale": 0.5}, {"time_scale": 1 / 3, "duration": 0.5}, {"start": 0.51},
                # The key frame found for 0.8 s starts after it, so ffmpeg loses frame 17.
                {"start": 0.8}, {"start": 0.8, "duration": 0.521, "time_scale": 0.7},
                {"start": 0.8, "duration": 0.12, "time_scale": 1 / 3}]
    if name in ("h264-jitter.mp4", "vp9-jitter.webm"):
        return [{}, {"time_scale": 0.5}, {"start": 0.51}, {"start": 0.51, "time_scale": 2 / 3}]
    if name == "h264-rot90-aac.mp4":
        return [{}, {"start": 0.51}, {"scale": 0.5, "time_scale": 0.5}]
    if name in ("h264-tmcd.mp4", "vp9-subs.webm"):
        # Starts over half a frame before the next frame: with a second stream of
        # any kind, ffmpeg repeats the first frame kept.
        return [{}, {"start": 0.51}, {"start": 0.523},
                {"start": 0.3, "duration": 0.4, "time_scale": 2}]
    if name == "vp9-opus.webm":
        # Starts within the Opus delay (6.5 ms) after a frame, which ffmpeg would
        # keep if the delay moved the file's start.
        return [{}, {"start": 0.523}, {"start": 0.526, "time_scale": 2},
                {"start": 0.51, "duration": 0.3}]
    # Starts whose key frame depends on the seek looking back by the reorder delay,
    # as it does under an edit list (here one starting the video late) and not without.
    if name == "h264-noedit.mp4":
        return [{}, {"start": 0.38}, {"start": 0.5393}, {"start": 0.8667, "time_scale": 1 / 3}]
    if name == "h264-vfr-late.mp4":
        return [{}, {"start": 0.62}, {"start": 0.63, "time_scale": 2},
                {"start": 0.625, "duration": 0.6, "time_scale": 0.5}]
    if name == "h264-empty-audio.mp4":
        # The video starts at its composition offset, and an audio track with no
        # packets has no start to move the file's back to 0.
        return [{}, {"start": 0.51}, {"time_scale": 0.5}, {"duration": 0.5}]
    if name in ("vp9-alpha.webm", "vp9-alpha-rot90.webm"):
        return [{}, {"scale": 0.5}, {"start": 0.51, "time_scale": 0.5}]
    if name in ("h264-tmcd-late.mp4", "vp9-subs-late.webm"):
        return [{}, {"time_scale": 0.5}, {"start": 0.51}]
    if name == "h264-tmcd-late15.mp4":
        return [{}]
    if name == "h264-bf2-vfr.mp4":
        # The real decode timestamps decide which frames ffmpeg repeats and drops.
        return [{}, {"time_scale": 0.5}, {"start": 0.51}, {"start": 0.89, "duration": 0.145}]
    if name == "h264-long-tail.mp4":
        return [{}, {"time_scale": 0.5}]
    if name in ("h264-vfr600.mp4", "h264-vfr600-240.mp4"):
        return [{"scale": 0.5}, {"scale": 0.5, "time_scale": 0.5},
                {"scale": 0.5, "start": 1.0, "duration": 1.0}]
    if name == "h264-vfr2400.mp4":
        return [{"scale": 0.5}, {"scale": 0.5, "time_scale": 0.5},
                {"scale": 0.5, "start": 0.1, "duration": 0.1}]
    if name == "h264-cut-aac.mp4":
        # Frames shown before the edit or from its end on are decoded but not output.
        return [{}, {"start": 0.1}, {"start": 0.2, "duration": 0.15},
                {"start": 0.12, "duration": 0.2, "time_scale": 2}]
    if name == "vp9-opus-stall.webm":
        # The frame after the stall starts late but lasts one frame, the duration ffmpeg
        # makes up for a block without one.
        return [{}, {"start": 0.1}, {"duration": 0.7}, {"time_scale": 0.5}]
    if name == "h264-ts60.mp4":
        return [{}, {"time_scale": 0.5}, {"start": 0.51}]
    if name == "h264-uneven-30.mp4":
        # Passing frames through, ffmpeg paces to the 30 fps declared rather than the 60 it
        # reports, which timeslice.probe can't see: only rates the fps filter sets are tried.
        return [{"time_scale": 0.5}, {"start": 0.51, "time_scale": 0.25}]
    if name in ("h264-uneven.mp4", "h264-uneven-frag.mp4", "hevc-uneven-30.mp4",
                "hevc-uneven-untimed.mp4"):
        return [{}, {"time_scale": 0.5}, {"start": 0.51, "duration": 0.5}]
    if name == "vp9-240-at-4.webm":
        return [{}, {"time_scale": 2}, {"start": 0.1}]
    if name == "vp9-uneven-frag.mp4":
        # Passed through, a frame without a duration lasts one at the rate paced to; before the
        # fps filter, the last lasts the step from the one before it.
        return [{}, {"time_scale": 2}, {"start": 0.51, "duration": 0.5}]
    if name in ("vp9-uneven-stall.mp4", "vp9-slow-stall.mp4", "vp9-600-stall.mp4"):
        return [{}, {"start": 0}]
    if name == "vp9-5-10.mp4":
        return [{}, {"start": 3.1}, {"duration": 7}]
    if name == "h264-30-at-10.mkv":
        return [{"time_scale": 0.5}, {"start": 0}]
    if name == "hevc-30-at-10.mkv":
        return [{}, {"time_scale": 0.5}, {"start": 0}]
    if name in ("vp9-5-10.webm", "hevc-untimed-5-10.mkv", "hevc-5-10.mkv"):
        return [{}, {"start": 3.1}]
    return [
        {},
        {"fast": True},
        {"scale": 0.5},
        {"scale": 0.5, "time_scale": 0.5, "fast": True},
        {"time_scale": 0.5},
        {"time_scale": 1 / 3},
        {"start": 0.5, "duration": 0.5},
        {"start": 0.51, "duration": 0.51},
        {"start": 0.51, "duration": 0.51, "time_scale": 0.5, "scale": 0.5, "fast": True},
        {"start": round(float(7.5 * frame), 4), "time_scale": 0.5},
        {"start": round(float(10.6 * frame), 4), "duration": round(float(9.4 * frame), 4),
         "time_scale": 0.5},
        # The last frame kept rounds down to a third-rate tick, its end rounds up.
        {"start": float(2 * frame), "time_scale": 1 / 3},
        {"duration": float(20 * frame), "time_scale": 1 / 3},
        # Within a tick of a frame: the window's ends are rounded to ticks.
        {"start": round(float(15 * frame + Fraction(2, 5) * tick), 6), "time_scale": 0.5},
        {"duration": round(float(20 * frame + Fraction(3, 5) * tick), 6)},
        {"duration": 0.7},
        {"start": 1.2},
        # ffmpeg's trim takes a duration of 0 to mean no limit.
        {"start": 1.2, "duration": 0},
        # Double rate repeats frames.
        {"duration": 0.2, "time_scale": 2},
        {"start": 0.51, "duration": 0.2, "time_scale": 2},
    ]


def identify(frame, references):
    """The index of the reference frame nearest `frame`, and how near it and
    the runner-up are (mean absolute difference per channel)."""
    diffs = np.abs(references.astype(np.int16) - frame.astype(np.int16)).mean(axis=(1, 2, 3))
    order = np.argsort(diffs)
    return int(order[0]), float(diffs[order[0]]), float(diffs[order[1]])


def quarter_turns(frame, source):
    """How many quarter turns anticlockwise take `source` to `frame`, one frame
    of a clip as load_video returned it at full size."""
    fits = sorted((np.abs(np.rot90(source, k).astype(np.int16) - frame).mean(), k)
                  for k in range(4) if np.rot90(source, k).shape == frame.shape)
    assert len(fits) == 1 or fits[1][0] > 20, fits
    return fits[0][1]


def references_for(frames, height, width, turns):
    """The source frames turned `turns` quarter turns anticlockwise and resized
    to height x width by nearest neighbour, as references to identify by."""
    turned = np.rot90(frames, turns, axes=(1, 2))
    rows = ((np.arange(height) + 0.5) * turned.shape[1] / height).astype(int)
    cols = ((np.arange(width) + 0.5) * turned.shape[2] / width).astype(int)
    return turned[:, rows][:, :, cols]


def video_error(call, path, name):
    """The message of the VideoError `call` raises, naming the file `name`."""
    try:
        call()
    except timeslice.VideoError as err:
        return str(err).replace(str(path), name)
    raise AssertionError(f"{name} loaded")


# Not a video at all, and a file holding only sound.
JUNK = "this is not a video\n" * 8
AUDIO = "audio.wav"
# MPEG-4 Part 2: ffmpeg decodes it, browsers do not.
UNDECODABLE = "mpeg4.mp4"
# An AVI file: ffmpeg reads it, browsers do not.
UNOPENABLE = "clip.avi"


def main():
    OUT.mkdir(exist_ok=True)
    frames = source_frames()
    clips, cases, pixels, offset = {}, [], [], 0
    for name, (rate, codec, pix_fmt, *steps) in CLIPS.items():
        path = encode(name, frames, rate, codec, pix_fmt, *steps)
        width, height, fps, duration = timeslice.probe(str(path))
        clips[name] = {"width": width, "height": height,
                       "fps": [fps.numerator, fps.denominator], "duration": duration}
        turns = quarter_turns(timeslice.load_video(str(path))[0][0], frames[0])
        for options in option_sets(name, rate, time_base(path)):
            volume, fps_out = timeslice.load_video(str(path), **options)
            _, h, w, _ = volume.shape
            references = references_for(frames, h, w, turns)
            indices = []
            for frame in volume:
                index, best, runner_up = identify(frame, references)
                assert best < 8 and runner_up > 15, (name, options, best, runner_up)
                indices.append(index)
            time_scale = options.get("time_scale", 1.0)
            kept = fps_out * Fraction(time_scale).limit_denominator(1000)
            cases.append({
                "clip": name,
                "options": {"scale": options.get("scale", 1.0), "timeScale": time_scale,
                            "start": options.get("start"), "duration": options.get("duration"),
                            "fast": options.get("fast", False)},
                "width": w, "height": h, "fps": [kept.numerator, kept.denominator],
                "frames": indices, "offset": offset,
            })
            pixels.append(volume.tobytes())
            offset += volume.nbytes

    clip = OUT / "h264-30.mp4"
    junk = OUT / "junk.mp4"
    junk.write_text(JUNK)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=0.05",
                    "-ar", "8000", *BITEXACT, str(OUT / AUDIO)], check=True)
    for name in (UNDECODABLE, UNOPENABLE):
        ffmpeg("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", "30",
               "-i", "-", "-c:v", "mpeg4", *COLOUR, "-pix_fmt", "yuv420p", *BITEXACT, OUT / name,
               input=frames[:2].tobytes())
        assert len(timeslice.load_video(str(OUT / name))[0]) == 2
    errors = [
        {"clip": "junk.mp4", "options": {},
         "message": video_error(lambda: timeslice.probe(str(junk)), junk, "junk.mp4")},
        {"clip": AUDIO, "options": {},
         "message": video_error(lambda: timeslice.probe(str(OUT / AUDIO)), OUT / AUDIO, AUDIO)},
        {"clip": clip.name, "options": {"start": 5},
         "message": video_error(lambda: timeslice.load_video(str(clip), start=5), clip, clip.name)},
    ]
    junk.unlink()
    seconds = [[length, start, duration, timeslice.clip_seconds(length, start, duration)]
               for length in (10, 1.5, None) for start in (None, 0, 0.25, 9, 12)
               for duration in (None, 0.5, 3, 20)]

    with open(OUT / "decode.json", "w") as out:
        out.write('{"junk": %s, "undecodable": "%s", "unopenable": "%s",\n'
                  % (json.dumps(JUNK), UNDECODABLE, UNOPENABLE))
        out.write(' "clips": %s,\n' % json.dumps(clips))
        out.write(' "errors": [\n  %s],\n' % ",\n  ".join(json.dumps(e) for e in errors))
        out.write(' "clipSeconds": %s,\n' % json.dumps(seconds))
        out.write(' "cases": [\n  %s]}\n' % ",\n  ".join(json.dumps(c) for c in cases))
    with gzip.GzipFile(OUT / "frames.bin.gz", "wb", compresslevel=9, mtime=0) as out:
        out.write(b"".join(pixels))


if __name__ == "__main__":
    main()
