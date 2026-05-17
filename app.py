import io
import logging
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from speech import SpeechService, SpeechConfig

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Speech-to-Text API",
    description="Une API professionnelle de transcription vocale temps réel et fichier basée sur Faster-Whisper.",
    version="0.1.0"
)

# Configuration de CORS universelle acceptant toutes les adresses IP et domaines (HTTP et HTTPS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement unique du service pour éviter la réinitialisation coûteuse du modèle Whisper à chaque requête
logger.info("Démarrage de l'API et chargement du service Speech...")
speech_service = SpeechService()
logger.info("Service Speech initialisé et prêt.")

# Modèle de données pour les réponses
class TranscriptionResponse(BaseModel):
    status: str
    transcription: str
    duration_processed: float = 0.0

@app.post(
    "/api/transcribe-file",
    response_model=TranscriptionResponse,
    summary="Transcrit un fichier audio (.wav)",
    description="Téléverse un fichier audio WAV (16kHz mono recommandé) et retourne sa transcription textuelle."
)
async def transcribe_file(file: UploadFile = File(...)):
    """
    Reçoit un fichier WAV par POST multipart/form-data.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .wav sont acceptés.")

    try:
        # Lire le fichier téléversé en mémoire
        file_bytes = await file.read()
        
        # Charger l'audio via SoundFile depuis les octets en mémoire
        audio_data, sample_rate = sf.read(io.BytesIO(file_bytes))
        
        # Convertir en mono si l'audio est en stéréo (Whisper requiert 1 canal)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        # Forcer le format float32 attendu par Faster-Whisper
        audio_data = audio_data.astype(np.float32)
        
        # Calcul de la durée du fichier audio original
        duration = len(audio_data) / sample_rate
        
        # Rééchantillonner à 16000Hz si le fichier d'entrée a un taux différent
        # (Whisper requiert strictement du 16000Hz pour transcrire correctement)
        if sample_rate != 16000:
            logger.info(f"Rééchantillonnage de l'audio de {sample_rate}Hz à 16000Hz.")
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), num_samples, endpoint=False),
                np.arange(len(audio_data)),
                audio_data
            ).astype(np.float32)
        
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier audio : {e}")
        raise HTTPException(status_code=400, detail=f"Erreur lors de la lecture du fichier audio : {str(e)}")

    # Lancement de la transcription
    try:
        text = speech_service.transcribe(audio_data)
        return TranscriptionResponse(
            status="success",
            transcription=text,
            duration_processed=round(duration, 2)
        )
    except Exception as e:
        logger.error(f"Erreur de transcription : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription : {str(e)}")

@app.post(
    "/api/record-and-transcribe",
    response_model=TranscriptionResponse,
    summary="Enregistre et transcrit en direct depuis le microphone",
    description="Déclenche l'écoute sur le microphone pour une durée fixe définie (en secondes). L'enregistrement se poursuit même en cas de silence."
)
async def record_and_transcribe(
    duration_s: int = Query(
        5,
        description="Durée de l'enregistrement en secondes (entre 1 et 30 secondes)",
        ge=1,
        le=30
    )
):
    """
    Enregistre l'audio du microphone pour une durée fixe, puis effectue sa transcription.
    """
    try:
        logger.info(f"Début de l'écoute fixe via l'API pour {duration_s} secondes...")
        text = speech_service.listen_fixed(duration_s)
        return TranscriptionResponse(
            status="success",
            transcription=text,
            duration_processed=float(duration_s)
        )
    except Exception as e:
        logger.error(f"Erreur lors de la capture audio fixe : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement ou de la transcription : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Lancement du serveur uvicorn sur le port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
