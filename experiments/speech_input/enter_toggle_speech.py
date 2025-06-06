import pyaudio
import wave
import threading
import keyboard  # pip install keyboard

filename = "recorded.wav"
chunk = 1024
fmt = pyaudio.paInt32
channels = 1
rate = 44100

frames = []
recording = False

def record():
    global frames, recording
    p = pyaudio.PyAudio()
    stream = p.open(format=fmt, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    print("🎙️ Recording... Press Enter again to stop.")

    while recording:
        data = stream.read(chunk)
        frames.append(data)

    print("🛑 Stopped recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(fmt))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"✅ Audio saved to {filename}")

def wait_for_enter_to_toggle():
    global recording, frames
    print("Press Enter to start recording...")
    while True:
        keyboard.wait('enter')
        if not recording:
            frames = []
            recording = True
            thread = threading.Thread(target=record)
            thread.start()
        else:
            recording = False
            break

if __name__ == "__main__":
    wait_for_enter_to_toggle()
    import whisper

    model = whisper.load_model("turbo")
    result = model.transcribe(filename)
    print("📝 Transcription:", result["text"])

