import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from speech import SpeechService

class TestSpeechService(unittest.TestCase):
    """
    Tests unitaires pour le service principal SpeechService.
    Utilisation de mocks pour isoler les dépendances et éviter le chargement 
    réel des modèles d'intelligence artificielle lors des tests rapides.
    """
    
    @patch('speech.service.VoiceActivityDetector')
    @patch('speech.service.AudioRecorder')
    @patch('speech.service.AudioTranscriber')
    def test_transcribe_direct_audio(self, MockTranscriber, MockRecorder, MockVAD):
        # Configuration des valeurs de retour du mock
        mock_transcriber_instance = MockTranscriber.return_value
        mock_transcriber_instance.transcribe.return_value = "Ceci est un test"
        
        # Instanciation du service (qui va utiliser nos mocks au lieu des vraies classes)
        service = SpeechService()
        
        # Simulation d'un fichier audio chargé en mémoire
        fake_audio = np.zeros(16000, dtype=np.float32)
        
        # Exécution de la méthode
        result = service.transcribe(fake_audio)
        
        # Vérification
        self.assertEqual(result, "Ceci est un test")
        mock_transcriber_instance.transcribe.assert_called_once_with(fake_audio)

if __name__ == '__main__':
    unittest.main()
