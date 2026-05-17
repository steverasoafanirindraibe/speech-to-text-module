import numpy as np
import soundfile as sf
import sys
from speech import SpeechService

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_file.py <chemin_vers_fichier_audio.wav>")
        return
        
    file_path = sys.argv[1]
    
    print(f"Chargement du fichier audio : {file_path}")
    try:
        # Utilisation de SoundFile pour charger l'audio
        # Pour Whisper, la fréquence d'échantillonnage doit être de 16kHz en mono.
        audio_data, sample_rate = sf.read(file_path)
        
        # Convertir en mono si le fichier est en stéréo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        # S'assurer du format float32
        audio_data = audio_data.astype(np.float32)
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return
        
    print("Initialisation du service Speech-to-Text...")
    speech = SpeechService()
    
    print("Lancement de la transcription...")
    text = speech.transcribe(audio_data)
    
    print("\n--- Résultat de la transcription ---")
    print(text)

if __name__ == "__main__":
    main()
