import torch
import numpy as np

class VoiceActivityDetector:
    """
    Responsable de la détection d'activité vocale (VAD).
    Évite les transcriptions inutiles de silence pour optimiser les performances.
    """
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        
        # Chargement du modèle Silero VAD depuis PyTorch Hub.
        # Le modèle est chargé en mémoire pour une exécution ultra-rapide par la suite.
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        # Désactivation de l'entraînement pour optimiser l'inférence
        self.model.eval()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Détermine si un segment audio contient de la parole.
        
        Args:
            audio_chunk: Tableau NumPy contenant les échantillons audio (float32).
            
        Returns:
            True si la probabilité de parole dépasse le seuil configuré.
        """
        if len(audio_chunk) == 0:
            return False
            
        # Le modèle Silero s'attend à des tenseurs PyTorch float32
        tensor_chunk = torch.from_numpy(audio_chunk).float()
        
        # Ajout de la dimension de batch si nécessaire
        if tensor_chunk.ndim == 1:
            tensor_chunk = tensor_chunk.unsqueeze(0)
            
        with torch.no_grad():
            speech_prob = self.model(tensor_chunk, self.sample_rate).item()
            
        return speech_prob >= self.threshold
