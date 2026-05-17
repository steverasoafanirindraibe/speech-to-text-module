import numpy as np
import time
import logging
from typing import Optional

from .config import SpeechConfig
from .recorder import AudioRecorder
from .vad import VoiceActivityDetector
from .transcriber import AudioTranscriber

# Configuration basique du logging pour l'ensemble du package
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechService:
    """
    Point d'entrée principal de l'API publique.
    Orchestre l'enregistrement, la détection vocale et la transcription.
    """
    def __init__(self, config: Optional[SpeechConfig] = None):
        """
        Initialise les composants requis (micro, VAD, IA) de manière centralisée.
        """
        self.config = config or SpeechConfig()
        
        # Instanciation modulaire des sous-systèmes
        self.recorder = AudioRecorder(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_duration_ms=self.config.chunk_duration_ms
        )
        
        self.vad = VoiceActivityDetector(
            sample_rate=self.config.sample_rate,
            threshold=self.config.vad_threshold
        )
        
        self.transcriber = AudioTranscriber(
            model_size=self.config.whisper_model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language
        )

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcrit un tableau audio direct en texte.
        Permet un traitement asynchrone ou la transcription de fichiers.
        
        Args:
            audio_data: Tableau Numpy 1D (float32) échantillonné à 16kHz.
            
        Returns:
            Texte transcrit.
        """
        return self.transcriber.transcribe(audio_data)

    def listen(self) -> str:
        """
        Écoute le microphone jusqu'à détecter un silence, puis transcrit la parole.
        Cette méthode est bloquante pendant la durée de l'écoute.
        
        Returns:
            Le texte transcrit de la phrase capturée.
        """
        self.recorder.start()
        
        audio_buffer = []
        is_speaking = False
        silence_start_time = None
        start_time = time.time()
        
        logger.info("En écoute...")
        
        try:
            for chunk in self.recorder.get_audio_stream():
                # Vérification de la limite de temps de sécurité globale
                if time.time() - start_time > self.config.max_record_duration_s:
                    logger.info("Limite de durée d'enregistrement maximale atteinte.")
                    break
                    
                # On utilise le VAD pour repérer le début de la voix
                chunk_has_speech = self.vad.is_speech(chunk)
                
                if chunk_has_speech:
                    if not is_speaking:
                        is_speaking = True
                    # Réinitialiser le compteur de silence
                    silence_start_time = None
                    audio_buffer.append(chunk)
                elif is_speaking:
                    # Ajouter le chunk même silencieux pour ne pas couper de respiration
                    audio_buffer.append(chunk)
                    
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    
                    # Arrêter l'écoute si le silence est trop long
                    if time.time() - silence_start_time > self.config.silence_timeout_s:
                        break
        except KeyboardInterrupt:
            logger.info("Interruption manuelle de l'écoute.")
        finally:
            self.recorder.stop()
            
        if not audio_buffer:
            return ""
            
        # Concaténer tous les fragments capturés en un seul tableau continu
        complete_audio = np.concatenate(audio_buffer)
        
        return self.transcriber.transcribe(complete_audio)

    def listen_fixed(self, duration_s: float) -> str:
        """
        Écoute le microphone pendant une durée fixe (en secondes) spécifiée,
        sans détection de silence ni VAD, puis transcrit le résultat.
        
        Args:
            duration_s: Durée fixe d'enregistrement en secondes.
            
        Returns:
            Le texte transcrit.
        """
        # S'assurer de ne pas dépasser la limite absolue de sécurité (ex. 30s)
        max_limit = max(self.config.max_record_duration_s, 30.0)
        duration_s = min(duration_s, max_limit)
        
        logger.info(f"Enregistrement fixe démarré pour {duration_s} secondes...")
        self.recorder.start()
        
        audio_buffer = []
        start_time = time.time()
        
        try:
            for chunk in self.recorder.get_audio_stream():
                audio_buffer.append(chunk)
                if time.time() - start_time >= duration_s:
                    break
        except KeyboardInterrupt:
            logger.info("Enregistrement interrompu manuellement.")
        finally:
            self.recorder.stop()
            
        if not audio_buffer:
            return ""
            
        # Concaténer tous les fragments
        complete_audio = np.concatenate(audio_buffer)
        return self.transcriber.transcribe(complete_audio)
