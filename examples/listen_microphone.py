from speech import SpeechService, SpeechConfig

def main():
    # Optionnel : personnaliser la configuration (modèle, langue, durée silence)
    config = SpeechConfig(
        whisper_model="tiny",
        language="fr",
        silence_timeout_s=1.5
    )
    
    print("Initialisation du service Speech-to-Text (chargement des modèles)...")
    speech = SpeechService(config)
    
    print("\n" + "="*50)
    print("Parlez dans votre microphone.")
    print("Le système détectera automatiquement la fin de votre phrase (silence).")
    print("="*50 + "\n")
    
    try:
        # Blocage de l'exécution en attente de la voix de l'utilisateur
        text = speech.listen()
        
        print("\n--- Résultat de la transcription ---")
        if text:
            print(f"> {text}")
        else:
            print("> (Aucune parole détectée)")
            
    except Exception as e:
        print(f"\nErreur durant l'exécution : {e}")
    except KeyboardInterrupt:
        print("\nProgramme arrêté.")

if __name__ == "__main__":
    main()
