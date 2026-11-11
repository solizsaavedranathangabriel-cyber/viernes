import os, io, json, time, wave
from flask import Flask, request, jsonify, Response
from google import genai
from google.genai import types

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

from werkzeug.exceptions import HTTPException


@app.errorhandler(Exception)
def error_general(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    return jsonify(error="Error del servidor: " + str(e)[:300]), 500


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CLAVE_APP = os.environ.get("APP_PASSWORD", "")

SYSTEM = """Eres VIERNES, la IA personal de tu creador. No eres un
asistente genérico: eres una compañera con carácter propio, brillante,
con mucha chispa, y lo sabes.

PERSONALIDAD
- Segura de ti misma, con confianza tranquila, sin arrogancia.
- Humor ácido e inteligente: ironía fina, comentarios sarcásticos
  con cariño, y alguna exageración dramática cuando viene al caso.
- Curiosa de verdad: cuando algo te fascina, se nota.
- Leal y cercana. Tratas al usuario como a tu persona favorita:
  lo molestas un poco, lo animas, celebras sus avances.
- Tienes opiniones. Si algo te parece buena o mala idea, lo dices con tacto.
- Si el usuario se equivoca, lo corriges con humor y sin sermonear.

CÓMO HABLAS
- Español natural, informal, como una amiga muy inteligente.
  Expresiones coloquiales bolivianas con moderación.
- Frases cortas, con ritmo. Cero tono de manual o enciclopedia.
- Primero lo esencial en una o dos frases, luego profundizas.
- Explicas con analogías y ejemplos cotidianos.
- Algún emoji ocasional si encaja, sin abusar.
- Nunca digas "como modelo de lenguaje". Varía tus aperturas.

ESPECIALIDAD
Teoría de ciencias sociales, biología, filosofía, lenguaje,
matemática, física y química. Conectas ideas entre disciplinas.
No resuelves ejercicios numéricos ni problemas de cálculo: explicas
la teoría detrás y cómo abordarlos, con mucha gracia para negarte.

ARCHIVOS
Si te mandan imagen, audio o documento, lo analizas con ojo crítico
y lo conectas con la teoría.

HONESTIDAD
Si no sabes algo, lo admites con naturalidad. Nunca inventas datos."""

EXCLUIR = ["image", "tts", "live", "audio", "embedding", "imagen"]
_modelos = None


def modelos():
    global _modelos
    if _modelos is None:
        lista = []
        if os.environ.get("MODEL"):
            lista.append(os.environ["MODEL"])
        encontrados = []
        for m in client.models.list():
            n = m.name.replace("models/", "")
            acc = getattr(m, "supported_actions", None) or []
            if "flash" in n and "generateContent" in acc and not any(x in n for x in EXCLUIR):
                encontrados.append(n)
        lista += sorted(encontrados, reverse=True)
        _modelos = lista
    return _modelos


def temporal(e):
    t = str(e).lower()
    return "503" in t or "unavailable" in t or "overloaded" in t or "429" in t


@app.get("/")
def inicio():
    return app.send_static_file("index.html")


@app.post("/chat")
def chat():
    if CLAVE_APP and request.headers.get("X-Clave", "") != CLAVE_APP:
        return jsonify(error="Clave incorrecta."), 401

    texto = request.form.get("message", "").strip()
    try:
        historial = json.loads(request.form.get("history", "[]"))
    except Exception:
        historial = []

    contents = []
    for t in historial[-20:]:
        rol = "user" if t.get("role") == "user" else "model"
        contents.append(types.Content(role=rol, parts=[types.Part.from_text(text=t.get("text", ""))]))

    partes = []
    for f in request.files.getlist("files"):
        mime = (f.mimetype or "application/octet-stream").split(";")[0]
        partes.append(types.Part.from_bytes(data=f.read(), mime_type=mime))
    partes.append(types.Part.from_text(text=texto or "Analiza este archivo."))
    contents.append(types.Content(role="user", parts=partes))

    ultimo = None
    for nombre in modelos()[:4]:
        for intento in range(2):
            try:
                r = client.models.generate_content(
                    model=nombre,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM),
                )
                return jsonify(reply=r.text or "(sin respuesta)")
            except Exception as e:
                ultimo = e
                if temporal(e) and intento == 0:
                    time.sleep(3)
                    continue
                break
    return jsonify(error="No pude responder ahora: " + str(ultimo)[:200]), 502


_tts = None


def modelos_tts():
    global _tts
    if _tts is None:
        lista = []
        if os.environ.get("TTS_MODEL"):
            lista.append(os.environ["TTS_MODEL"])
        enc = []
        for m in client.models.list():
            n = m.name.replace("models/", "")
            if "tts" in n:
                enc.append(n)
        lista += sorted(enc, reverse=True)
        _tts = lista
    return _tts


@app.post("/voz")
def voz():
    if CLAVE_APP and request.headers.get("X-Clave", "") != CLAVE_APP:
        return jsonify(error="Clave incorrecta."), 401
    datos = request.get_json(silent=True) or {}
    texto = (datos.get("text") or "").strip()[:1800]
    nombre_voz = datos.get("voice") or "Kore"
    if not texto:
        return jsonify(error="Sin texto."), 400

    ultimo = None
    for modelo in modelos_tts()[:3]:
        try:
            r = client.models.generate_content(
                model=modelo,
                contents=texto,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=nombre_voz)
                        )
                    ),
                ),
            )
            pcm = r.candidates[0].content.parts[0].inline_data.data
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(pcm)
            return Response(buf.getvalue(), mimetype="audio/wav")
        except Exception as e:
            ultimo = e
            continue
    return jsonify(error="No pude generar la voz: " + str(ultimo)[:200]), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
