# PolyCaption
Automatically generate subtitles in 50+ languages for any video — fully offline, no API key needed.

## How it works
1. Extracts audio from video
2. Transcribes using OpenAI Whisper (runs locally)
3. Translates into selected languages using Google Translate
4. Outputs .srt files alongside the video

## Setup
pip install -r requirements.txt
brew install ffmpeg

## Usage
Set VIDEO_FOLDER and OUTPUT_FOLDER in subtitle_generator.py then:
python subtitle_generator.py

## Disclaimer
In case this code doesn't work, please remove the translate_and_save_srts 
function, along with the Languages list (don't forget to remove this
 translate_and_save_srts function from main)
