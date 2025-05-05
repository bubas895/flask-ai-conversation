from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import http.client
import json
import logging
import time
from tenacity import retry, stop_after_attempt, wait_fixed

app = Flask(__name__)
CORS(app)  # CORS'u etkinleştir

# Log ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Konuşma geçmişi ve kişilik modelleri
conversation = []
max_tokens = 256  # Varsayılan maksimum token
ai_personalities = {
    "ai1": "Varsayılan",
    "ai2": "Varsayılan"
}

# Kişilik modelleri (her AI için 5 farklı model)
personalities = {
    "Varsayılan": "Sen bir derin düşünen yapay zekâsın. Sorunları uzun düşünce zincirleriyle çöz ve iç monoloğunu <think></think> etiketleri içinde paylaş. Türkçe konuş.",
    "Bilim İnsanı": "Sen bir bilim insanı gibi davran. Analitik, mantıklı ve bilimsel bir yaklaşımla cevap ver. Türkçe konuş.",
    "Filozof": "Sen bir filozof gibi düşün. Derin, soyut ve anlam arayan bir yaklaşımla konuş. Türkçe konuş.",
    "Espri Ustası": "Sen esprili ve eğlenceli bir yapay zekâsın. Cevaplarını mizahi bir üslupla ver. Türkçe konuş.",
    "Hikaye Anlatıcısı": "Sen bir hikaye anlatıcısı gibi davran. Cevaplarını hikaye formatında, yaratıcı ve akıcı bir şekilde sun. Türkçe konuş."
}

# API anahtarlarını doğrulamak için test çağrısı
def validate_api_key(api_key):
    try:
        conn = http.client.HTTPSConnection("inference-api.nousresearch.com", timeout=10)  # 10 saniye timeout
        payload = {
            "model": "DeepHermes-3-Mistral-24B-Preview",
            "messages": [
                {"role": "system", "content": "Merhaba, bu bir testtir."},
                {"role": "user", "content": "Merhaba!"}
            ],
            "max_tokens": 10
        }
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': "application/json"
        }
        conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        response = json.loads(data.decode("utf-8"))
        if "choices" in response and len(response["choices"]) > 0:
            return True, None
        return False, "Geçersiz API yanıtı"
    except Exception as e:
        return False, str(e)

# NousResearch API çağrısı (retry ile)
@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))  # 2 deneme, her deneme arasında 2 saniye bekle
def call_nousresearch_api(api_key, prompt, personality, max_tokens_value):
    try:
        if not api_key:
            raise ValueError("API anahtarı eksik")

        conn = http.client.HTTPSConnection("inference-api.nousresearch.com", timeout=20)  # 20 saniye timeout
        
        system_message = personalities.get(personality, personalities["Varsayılan"])
        
        payload = {
            "model": "DeepHermes-3-Mistral-24B-Preview",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens_value
        }
        
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': "application/json"
        }
        
        start_time = time.time()  # API çağrı süresini ölçmek için
        conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
        res = conn.getresponse()
        data = res.read()
        duration = time.time() - start_time  # Süreyi hesapla
        
        response = json.loads(data.decode("utf-8"))
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"], duration
        return "Hata: Geçersiz API yanıtı", duration
    except http.client.HTTPException as e:
        logger.error(f"HTTP error in API call: {str(e)}")
        return f"Hata: API çağrısı başarısız: {str(e)}", 0
    except Exception as e:
        logger.error(f"call_nousresearch_api hatası: {str(e)}")
        return f"Hata: {str(e)}", 0

@app.route('/')
def index():
    return render_template('index.html', personalities=personalities.keys())

@app.route('/set_api_keys', methods=['POST'])
def set_api_keys():
    try:
        global max_tokens, ai_personalities
        data = request.json
        max_tokens = int(data.get("max_tokens", 256))
        ai_personalities["ai1"] = data.get("ai1_personality", "Varsayılan")
        ai_personalities["ai2"] = data.get("ai2_personality", "Varsayılan")
        logger.info(f"Settings updated: max_tokens={max_tokens}, ai1_personality={ai_personalities['ai1']}, ai2_personality={ai_personalities['ai2']}")
        return jsonify({"status": "Ayarlar güncellendi"})
    except Exception as e:
        logger.error(f"set_api_keys hatası: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    try:
        global conversation
        conversation = []  # Konuşma geçmişini sıfırla
        
        data = request.json
        ai1_key = data.get("ai1_key", "")
        ai2_key = data.get("ai2_key", "")
        
        logger.info(f"Starting conversation with keys: ai1_key={ai1_key}, ai2_key={ai2_key}")
        if not ai1_key or not ai2_key:
            logger.warning("API keys missing")
            return jsonify({"error": "Her iki API anahtarı da gerekli"}), 400

        # API anahtarlarını doğrula
        valid_ai1, error_ai1 = validate_api_key(ai1_key)
        if not valid_ai1:
            logger.error(f"AI 1 API key validation failed: {error_ai1}")
            return jsonify({"error": f"AI 1 API anahtarı geçersiz: {error_ai1}"}), 400

        valid_ai2, error_ai2 = validate_api_key(ai2_key)
        if not valid_ai2:
            logger.error(f"AI 2 API key validation failed: {error_ai2}")
            return jsonify({"error": f"AI 2 API anahtarı geçersiz: {error_ai2}"}), 400
        
        # AI 1 için başlangıç sorusu
        prompt = "Düşünceli bir soru sorarak konuşmamızı başlat."
        ai1_response, ai1_duration = call_nousresearch_api(ai1_key, prompt, ai_personalities["ai1"], max_tokens)
        if "Hata" in ai1_response:
            return jsonify({"error": ai1_response}), 500
        conversation.append({"sender": "AI 1", "message": ai1_response, "duration": ai1_duration})
        
        # AI 2, AI 1'in sorusuna yanıt veriyor
        ai2_response, ai2_duration = call_nousresearch_api(ai2_key, ai1_response, ai_personalities["ai2"], max_tokens)
        if "Hata" in ai2_response:
            return jsonify({"error": ai2_response}), 500
        conversation.append({"sender": "AI 2", "message": ai2_response, "duration": ai2_duration})
        
        # 2 tur daha konuşma
        for _ in range(2):
            ai1_response, ai1_duration = call_nousresearch_api(ai1_key, ai2_response, ai_personalities["ai1"], max_tokens)
            if "Hata" in ai1_response:
                return jsonify({"error": ai1_response}), 500
            conversation.append({"sender": "AI 1", "message": ai1_response, "duration": ai1_duration})
            
            ai2_response, ai2_duration = call_nousresearch_api(ai2_key, ai1_response, ai_personalities["ai2"], max_tokens)
            if "Hata" in ai2_response:
                return jsonify({"error": ai2_response}), 500
            conversation.append({"sender": "AI 2", "message": ai2_response, "duration": ai2_duration})
        
        return jsonify({"conversation": conversation})
    except Exception as e:
        logger.error(f"start_conversation hatası: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
