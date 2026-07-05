import os
import math
import time
import whisper
from pydub import AudioSegment


LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh-CN": "Chinese",
    "ar": "Arabic",
}


VIDEO_FOLDER  = "VIDEO-PATH"            # as input
OUTPUT_FOLDER = "OUTPUT-PATH"           # as output
SRT_FOLDER    = "SRT-FILES-PATH"        # as output
CHUNK_MINS    = 8                       # may vary 
WHISPER_MODEL = "small"                 # Accuracy level(time required ) - large > medium > small > base


def load_model():
    print(f"\n Loading Whisper '{WHISPER_MODEL}' model...")
    model = whisper.load_model(WHISPER_MODEL)
    print(f" Model loaded Successfully \n")
    return model


def get_video_files():
    files = [
        os.path.join(VIDEO_FOLDER, f)
        for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(".mp4")
    ]
    return sorted(files)


def extract_audio(video_path, audio_path="temp_audio.mp3"):
    print(f"\n   [1/4] Extracting audio...")
    print(f"         from: {os.path.basename(video_path)}")
    os.system(
        f'ffmpeg -i "{video_path}" '
        f'-ar 16000 -ac 1 -b:a 32k '
        f'"{audio_path}" -y -loglevel quiet'
    )
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"         audio size: {size_mb:.1f}MB completed")
    return audio_path


def split_audio(audio_path):
    print(f"\n   [2/4] Splitting audio into chunks...")
    chunk_duration_ms = CHUNK_MINS * 60 * 1000
    audio             = AudioSegment.from_file(audio_path)
    total_chunks      = math.ceil(len(audio) / chunk_duration_ms)

    os.makedirs("temp_chunks", exist_ok=True)
    chunks = []

    print(f"         duration : {len(audio)/60000:.1f} mins")
    print(f"         chunks   : {total_chunks}")

    for i in range(total_chunks):
        start      = i * chunk_duration_ms
        end        = min((i + 1) * chunk_duration_ms, len(audio))
        chunk_path = f"temp_chunks/chunk_{i}.mp3"
        audio[start:end].export(chunk_path, format="mp3")
        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        print(f"         chunk {i+1}/{total_chunks}: {size_mb:.1f}MB")
        chunks.append((chunk_path, start))

    return chunks


def transcribe_chunks(chunks, model):
    print(f"\n   [3/4] Transcribing...")
    all_segments = []

    for i, (chunk_path, offset_ms) in enumerate(chunks):
        print(f"         chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
        start_time = time.time()
        result     = model.transcribe(chunk_path)
        elapsed    = time.time() - start_time
        print(f" {len(result['segments'])} segments ({elapsed:.0f}s)")

        for seg in result["segments"]:
            all_segments.append({
                "start": seg["start"] + offset_ms / 1000,
                "end":   seg["end"]   + offset_ms / 1000,
                "text":  seg["text"].strip()
            })

    print(f"         total segments: {len(all_segments)}")
    return all_segments


def translate_and_save_srts(segments, output_path, episode_num):
    from deep_translator import GoogleTranslator

    for lang_code, lang_name in LANGUAGES.items():
        print(f"         translating to {lang_name}...", end=" ", flush=True)

        translated_segments = []
        for seg in segments:
            try:
                translated_text = GoogleTranslator(
                    source="auto",
                    target=lang_code
                ).translate(seg["text"])
            except Exception:
                translated_text = seg["text"]  # Rollback

            translated_segments.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  translated_text
            })

       
        srt_path = os.path.join(OUTPUT_FOLDER, f"ep{episode_num}.{lang_code}.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(translated_segments, 1):
                f.write(f"{i}\n")
                f.write(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")

        

    print(f"         all languages saved successfully")


def seconds_to_srt_time(s):
    hrs  = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    ms   = int((s - int(s)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


def generate_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(seg['start'])} --> {seconds_to_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    print(f"         srt saved: {os.path.basename(srt_path)}")
    return srt_path


def burn_subtitles(video_path, srt_path, output_path):
    print(f"\n   [4/4] Embedding subtitles into video...")

    result = os.system(
        f'ffmpeg -i "{video_path}" -i "{srt_path}" '
        f'-c copy -c:s mov_text '
        f'"{output_path}" -y -loglevel quiet'
    )

    if not os.path.exists(output_path):
        raise RuntimeError(f"FFmpeg failed — exit code {result}")

    
    import shutil
    srt_alongside = output_path.replace(".mp4", ".srt")
    shutil.copy(srt_path, srt_alongside)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"         output: {os.path.basename(output_path)} ({size_mb:.0f}MB) ✅")


def cleanup(audio_path, chunks):
    import shutil
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists("temp_chunks"):
        shutil.rmtree("temp_chunks")  


DONE_LOG = os.path.join(VIDEO_FOLDER, "done.txt")

def already_done(video_path):
    if not os.path.exists(DONE_LOG):
        return False
    return os.path.basename(video_path) in open(DONE_LOG).read()

def mark_done(video_path):
    with open(DONE_LOG, "a") as f:
        f.write(os.path.basename(video_path) + "\n")


def main():
    print("=" * 55)
    print("           SUBTITLE GENERATOR — OFFLINE")
    print("=" * 55)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(SRT_FOLDER,    exist_ok=True)

    model       = load_model()
    video_files = get_video_files()

    print(f"Found {len(video_files)} video(s):\n")
    for i, v in enumerate(video_files, 1):
        size_mb = os.path.getsize(v) / (1024 * 1024)
        done    = "✅ done" if already_done(v) else "⏳ pending"
        print(f"   {i}. {os.path.basename(v)} ({size_mb:.0f}MB) — {done}")

    for idx, video_path in enumerate(video_files):
        print(f"\n{'='*55}")
        print(f"  VIDEO {idx+1}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*55}")

        if already_done(video_path):
            print("  ⏭️  Already processed — skipping")
            continue

        output_path = os.path.join(OUTPUT_FOLDER, f"ep{idx+1}.mp4")
        srt_path    = os.path.join(SRT_FOLDER,    f"ep{idx+1}.srt")

        try:
            audio_path = extract_audio(video_path)
            chunks     = split_audio(audio_path)
            segments   = transcribe_chunks(chunks, model)
            generate_srt(segments, srt_path)
            burn_subtitles(video_path, srt_path, output_path)
            translate_and_save_srts(segments, output_path, idx+1)
            cleanup(audio_path, chunks)
            mark_done(video_path)
            print(f"\n  ✅ VIDEO {idx+1} COMPLETE → ep{idx+1}.mp4")

        except Exception as e:
            print(f"\n  !!! ERROR on video {idx+1}: {e}")
            print("     Skipping to next video...")
            continue

    print(f"\n{'='*55}")
    print(f"   ALL DONE!")
    print(f"  Subtitled videos saved in: {OUTPUT_FOLDER}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()