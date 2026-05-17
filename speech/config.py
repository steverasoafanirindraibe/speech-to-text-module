from dataclasses import dataclass
import torch

@dataclass
class SpeechConfig:
    """
    Configuration globale du module de reconnaissance vocale.
    Centralise tous les paramètres pour faciliter la maintenance et l'injection.
    """
    # Paramètres audio généraux
    sample_rate: int = 16000
    channels: int = 1
    
    # Paramètres d'enregistrement
    chunk_duration_ms: int = 100
    silence_timeout_s: float = 2.0
    
    # Paramètres Faster-Whisper
    whisper_model: str = "tiny"
    language: str = "fr"
    # Utiliser le GPU si disponible, sinon CPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # L'utilisation de int8 offre un excellent compromis performance/mémoire
    compute_type: str = "int8"
    
    # Paramètres VAD (Silero)
    vad_threshold: float = 0.5
