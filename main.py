import os
import logging
from faster_whisper import WhisperModel
import datetime
import subprocess

# Constants
AUDIO_FILE_NAME = "./media/中間発表練習, 2024年8月31日 - 13-21-14.mp4"
MODEL_SIZE = "large-v3"
OUTPUT_DIR = "/app/mount"

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def is_cuda_available():
    try:
        # nvidia-smi コマンドを実行
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_device_and_compute_type():
    if is_cuda_available():
        return "cuda", "float16"
    else:
        return "cpu", "int8"

def load_model(model_size, device, compute_type):
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_audio(model, audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")
    
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        vad_filter=True,
        without_timestamps=True,
    )
    return segments, info

def write_transcription(segments, output_file):
    with open(output_file, "w") as f:
        for segment in segments:
            logger.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            f.write(f"{segment.text}\n")

def main():    
    try:
        device, compute_type = get_device_and_compute_type()
        logger.info(f"Using device: {device}, compute type: {compute_type}")
        
        model = load_model(MODEL_SIZE, device, compute_type)
        segments, info = transcribe_audio(model, AUDIO_FILE_NAME)
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"transcription_{timestamp}.txt")
        
        write_transcription(segments, output_file)
        logger.info(f"Transcription saved to {output_file}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()