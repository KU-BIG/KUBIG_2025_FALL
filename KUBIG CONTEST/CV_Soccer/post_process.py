#!/usr/bin/env python3
import os, sys, json, subprocess, tempfile, shutil, math

def ffprobe_duration(path):
    cmd = [
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1", path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def run(cmd):
    subprocess.run(cmd, check=True)

def main():
    if len(sys.argv) < 4:
        print("Usage: make_highlights.py input.mp4 segments.json output.mp4")
        sys.exit(1)

    video, seg_json, out = sys.argv[1:4]
    V_CODEC = os.environ.get("V_CODEC", "libx264")
    V_CRF   = os.environ.get("V_CRF", "20")
    V_PRE   = os.environ.get("V_PRESET", "veryfast")
    A_CODEC = os.environ.get("A_CODEC", "aac")
    A_BR    = os.environ.get("A_BITRATE", "128k")

    dur = round(ffprobe_duration(video), 3)
    head = (0.0, min(15.0, dur))
    tail = (max(dur-15.0, 0.0), dur)

    with open(seg_json, "r") as f:
        segs = json.load(f)

    # (start, end, label) 리스트
    merged = [ (head[0], head[1], "intro") ]
    for s in segs:
        start = float(s["start"]); end = float(s["end"])
        label = str(s.get("class","event"))
        if end > start:
            merged.append( (start, end, label) )
    merged.append( (tail[0], tail[1], "outro") )

    workdir = tempfile.mkdtemp()
    try:
        concat_path = os.path.join(workdir, "concat.txt")
        with open(concat_path, "w") as cf:
            for i, (s, e, label) in enumerate(merged):
                clip = os.path.join(workdir, f"clip_{i:03d}_{label}.mp4")
                cmd = [
                    "ffmpeg","-hide_banner","-loglevel","error",
                    "-ss", f"{s}", "-to", f"{e}", "-i", video,
                    "-c:v", V_CODEC, "-preset", V_PRE, "-crf", V_CRF,
                    "-c:a", A_CODEC, "-b:a", A_BR,
                    "-y", clip
                ]
                run(cmd)
                cf.write(f"file '{clip}'\n")

        # 최종 병합 (모든 클립 동일 코덱 → copy)
        run([
            "ffmpeg","-hide_banner","-loglevel","error",
            "-f","concat","-safe","0","-i", concat_path,
            "-c","copy","-y", out
        ])
        print(f"[Done] {out}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    main()
