import numpy as np
import sounddevice as sd
import queue
from typing import Generator

class AudioRecorder:
    """
    Responsable uniquement de la capture audio depuis le microphone.
    Ne contient aucune logique de transcription ou de VAD.
    """
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_duration_ms: int = 100):
        self.sample_rate = sample_rate
        self.channels = channels
        # Calcul du nombre de frames par bloc (chunk)
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback appelé par SoundDevice pour chaque bloc audio."""
        if status:
            # En production, on pourrait ajouter un logging spécifique des erreurs audio
            pass
            
        # Aplatir le tableau et copier pour éviter les problèmes de référence
        # avec le buffer interne de SoundDevice
        self.audio_queue.put(indata.flatten().copy())

    def start(self) -> None:
        """Démarre l'enregistrement audio en arrière-plan."""
        if self.is_recording:
            return
            
        self.is_recording = True
        
        # Vider la file d'attente avant de commencer un nouvel enregistrement
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
            
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self) -> None:
        """Arrête l'enregistrement audio."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio_stream(self) -> Generator[np.ndarray, None, None]:
        """Générateur fournissant l'audio capturé en continu."""
        while self.is_recording:
            try:
                # Utilisation d'un timeout pour permettre de vérifier 
                # la variable is_recording régulièrement
                data = self.audio_queue.get(timeout=0.1)
                yield data
            except queue.Empty:
                continue
