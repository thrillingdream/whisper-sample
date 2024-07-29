import os
import logging
from faster_whisper import WhisperModel
import os
import datetime

# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Define a handler to output log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Define a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# Add the console handler to the logger
logger.addHandler(console_handler)


AUDIO_FILE_NAME = "./audio/20240715.m4a"

## GPU付デバイスの場合
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

## CPUの場合
# model = WhisperModel("large-v3", device="cpu", compute_type="int16")

assert os.path.exists(AUDIO_FILE_NAME), "Audio file does not exist"

segments, info = model.transcribe(
	AUDIO_FILE_NAME,
	beam_size=5,
	vad_filter=True,
	without_timestamps=True,
)

logger.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Open a text file for writing with timestamp in the filename
output_file = open(f"/app/mount/transcription_{timestamp}.txt", "w")

for segment in segments:
	logger.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
	# Write the segment text to the file
	output_file.write(segment.text + "\n")

# Close the file
output_file.close()
