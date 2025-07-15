from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import logging
from google.cloud import texttospeech
from google.oauth2 import service_account
import tempfile
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google Cloud Text-to-Speech client using environment variables
def create_tts_client():
    try:
        # Create credentials from environment variables
        credentials_info = {
            "type": "service_account",
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
            "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID"),
            "private_key": (os.getenv("GOOGLE_CLOUD_PRIVATE_KEY") or "").replace('\\n', '\n'),
            "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
            "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_x509_CERT_URL"),
            "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_x509_CERT_URL"),
            "universe_domain": os.getenv("GOOGLE_UNIVERSE_DOMAIN")
        }
        
        # Validate required environment variables
        required_vars = [
            "GOOGLE_CLOUD_PROJECT_ID",
            "GOOGLE_CLOUD_PRIVATE_KEY",
            "GOOGLE_CLOUD_CLIENT_EMAIL"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return None
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        
        # Create TTS client with credentials
        client = texttospeech.TextToSpeechClient(credentials=credentials)
        logger.info("Google Cloud Text-to-Speech client initialized successfully from environment variables")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
        return None

# Initialize the client
client = create_tts_client()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Flask TTS API',
        'tts_available': client is not None
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech using Google Cloud Text-to-Speech API
    
    Expected JSON payload:
    {
        "text": "Hello world",
        "languageCode": "en-US",
        "voiceName": "en-US-Wavenet-D"
    }
    """
    try:
        # Check if TTS client is available
        if client is None:
            return jsonify({
                'error': 'Text-to-Speech service is not available. Please check your Google Cloud configuration.'
            }), 503

        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract required fields
        text = data.get('text', '').strip()
        language_code = data.get('languageCode', 'en-US')
        voice_name = data.get('voiceName', 'en-US-Wavenet-D')
        
        # Validate input
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        logger.info(f"TTS request - Language: {language_code}, Voice: {voice_name}, Text length: {len(text)}")
        
        # Configure the synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure the voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        # Configure the audio format
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0,
            volume_gain_db=0.0
        )
        
        # Make the TTS request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(response.audio_content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"TTS synthesis completed successfully for language: {language_code}")
        
        # Return the audio file
        return send_file(
            tmp_file_path,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name='speech.mp3'
        )
        
    except Exception as e:
        logger.error(f"TTS Error: {str(e)}")
        
        # Handle specific Google Cloud errors
        if "PERMISSION_DENIED" in str(e):
            return jsonify({
                'error': 'Permission denied. Please check your Google Cloud credentials and API access.'
            }), 403
        elif "INVALID_ARGUMENT" in str(e):
            return jsonify({
                'error': f'Invalid request parameters: {str(e)}'
            }), 400
        elif "UNIMPLEMENTED" in str(e):
            return jsonify({
                'error': 'TTS service is currently unavailable. Please try again later.'
            }), 503
        else:
            return jsonify({
                'error': f'TTS service error: {str(e)}'
            }), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    """List available voices for TTS"""
    try:
        if client is None:
            return jsonify({
                'error': 'Text-to-Speech service is not available'
            }), 503
        
        # Get language code from query parameters
        language_code = request.args.get('language_code', '')
        
        # List available voices
        voices = client.list_voices(language_code=language_code)
        
        voice_list = []
        for voice in voices.voices:
            voice_list.append({
                'name': voice.name,
                'language_codes': list(voice.language_codes),
                'ssml_gender': voice.ssml_gender.name,
                'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
            })
        
        return jsonify({
            'voices': voice_list,
            'total_count': len(voice_list)
        })
        
    except Exception as e:
        logger.error(f"Error listing voices: {str(e)}")
        return jsonify({'error': f'Failed to list voices: {str(e)}'}), 500

@app.route('/supported-languages', methods=['GET'])
def supported_languages():
    """Return list of supported languages"""
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'nl': 'Dutch',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'pl': 'Polish',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'cs': 'Czech',
        'sk': 'Slovak',
        'hu': 'Hungarian',
        'ro': 'Romanian',
        'bg': 'Bulgarian',
        'hr': 'Croatian',
        'sl': 'Slovenian',
        'et': 'Estonian',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tl': 'Filipino',
        'el': 'Greek',
        'he': 'Hebrew',
        'bn': 'Bengali',
        'ta': 'Tamil',
        'te': 'Telugu',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'pa': 'Punjabi',
        'ur': 'Urdu',
        'sw': 'Swahili',
        'af': 'Afrikaans',
        'is': 'Icelandic'
    }
    
    return jsonify({
        'languages': languages,
        'total_count': len(languages)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    required_env_vars = [
        "GOOGLE_CLOUD_PROJECT_ID",
        "GOOGLE_CLOUD_PRIVATE_KEY",
        "GOOGLE_CLOUD_CLIENT_EMAIL"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        for var in missing_vars:
            logger.info(f"  {var}")
    else:
        logger.info("All required environment variables are set")
    
    print("Starting Flask TTS API server...")
    print("Server will run at: http://localhost:8000")
    print("Available endpoints:")
    print("  POST /tts - Convert text to speech")
    print("  GET  /voices - List available voices")
    print("  GET  /supported-languages - Get supported languages")
    print("  GET  /health - Health check")
    
    port = int(os.environ["PORT"])  # <- no fallback
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )

