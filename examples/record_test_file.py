import sounddevice as sd
import soundfile as sf
import numpy as np
import time

def main():
    sample_rate = 16000
    duration = 5.0  # Durée de l'enregistrement en secondes
    filename = "test_simulation.wav"
    
    print("="*50)
    print(f"Préparation de l'enregistrement de {duration} secondes...")
    print("Préparez-vous à parler (ex: 'Bonjour, ceci est un test de transcription').")
    print("="*50 + "\n")
    
    for i in range(3, 0, -1):
        print(f"Démarrage dans {i}...")
        time.sleep(1)
        
    print("\n🔴 ENREGISTREMENT EN COURS...")
    
    # Enregistrer l'audio en 16kHz mono float32
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()  # Attendre que l'enregistrement se termine
    
    print("🟢 ENREGISTREMENT TERMINÉ !")
    
    # Sauvegarder l'audio dans un fichier .wav
    sf.write(filename, audio, sample_rate)
    print(f"\nFichier audio sauvegardé sous : {filename}")
    print(f"Vous pouvez maintenant le transcrire avec la commande suivante :")
    print(f"python examples/transcribe_file.py {filename}")

if __name__ == "__main__":
    main()
