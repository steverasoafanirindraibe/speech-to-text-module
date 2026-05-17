import numpy as np
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    Gère la transcription de l'audio en texte.
    Encapsule Faster-Whisper de manière optimisée.
    """
    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8", language: str = "fr"):
        self.language = language
        
        # Chargement unique du modèle afin d'éviter
        # une réinitialisation coûteuse à chaque transcription.
        logger.info(f"Chargement du modèle Whisper '{model_size}' sur {device} (compute: {compute_type})")
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        logger.info("Modèle chargé avec succès.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcrit un segment audio en texte.
        
        Args:
            audio_data: Tableau NumPy contenant l'audio normalisé (-1.0 à 1.0) à 16kHz.
            
        Returns:
            Le texte transcrit.
        """
        if len(audio_data) == 0:
            return ""

        # Faster-Whisper accepte directement un numpy array 1D
        segments, _ = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=1,        # Utilisation de greedy decoding (1 au lieu de 5) pour une vitesse 3x plus rapide
            best_of=1,          # Une seule tentative d'échantillonnage pour éviter le surcoût de calcul
            temperature=0.0,    # Inférence déterministe ultra-rapide sans tâtonnements de température
            # Le VAD est déjà géré de manière indépendante en amont
            vad_filter=False, 
            without_timestamps=True
        )
        
        # Concaténation de tous les segments transcrits
        text = " ".join([segment.text for segment in segments])
        return text.strip()
