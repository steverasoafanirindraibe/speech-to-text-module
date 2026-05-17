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
        Découpe automatiquement l'audio en fenêtres de taille requise par Silero VAD
        (512 échantillons à 16kHz, 256 à 8kHz) pour éviter les plantages de TorchScript.
        
        Args:
            audio_chunk: Tableau NumPy contenant les échantillons audio (float32).
            
        Returns:
            True si la probabilité de parole dépasse le seuil configuré sur l'une des fenêtres.
        """
        if len(audio_chunk) == 0:
            return False
            
        # Détermination de la taille de fenêtre requise par le modèle Silero
        window_size = 512 if self.sample_rate == 16000 else 256
        
        # Si le fragment global est trop court, on le remplit de zéros
        if len(audio_chunk) < window_size:
            padding = np.zeros(window_size - len(audio_chunk), dtype=np.float32)
            audio_chunk = np.concatenate([audio_chunk, padding])
            
        # Découpage et évaluation de chaque sous-fenêtre
        for i in range(0, len(audio_chunk), window_size):
            chunk = audio_chunk[i:i + window_size]
            
            # Gérer la dernière fenêtre incomplète si nécessaire
            if len(chunk) < window_size:
                padding = np.zeros(window_size - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, padding])
                
            tensor_chunk = torch.from_numpy(chunk).float()
            if tensor_chunk.ndim == 1:
                tensor_chunk = tensor_chunk.unsqueeze(0)
                
            with torch.no_grad():
                speech_prob = self.model(tensor_chunk, self.sample_rate).item()
                
            if speech_prob >= self.threshold:
                return True
                
        return False
