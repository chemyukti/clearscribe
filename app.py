"""ClearScribe web app (Flask).

Run:  python app.py    then open http://127.0.0.1:5000
"""

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from clearscribe import config
from clearscribe.audio import load_audio
from clearscribe.glossary import GlossaryError, final_text, parse_glossary
from clearscribe.pipeline import transcribe


def create_app(model_loader=None, upload_dir=None) -> Flask:
    """``model_loader`` returns the speech model; defaults to the wav2vec 2.0 model in config."""
    if model_loader is None:
        from clearscribe.asr import get_model as model_loader

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    uploads = Path(upload_dir or os.environ.get("CLEARSCRIBE_UPLOAD_DIR")
                   or Path(tempfile.gettempdir()) / "clearscribe-uploads")
    uploads.mkdir(parents=True, exist_ok=True)

    # Jobs live only for the lifetime of the server: each glossary belongs to one upload.
    jobs = {}
    lock = threading.Lock()

    def get_job(job_id):
        with lock:
            job = jobs.get(job_id)
        if job is None:
            abort(404, description="Unknown job.")
        return job

    def error(message, status=400):
        return jsonify({"error": message}), status

    @app.errorhandler(413)
    def too_large(_):
        return error(f"The file is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            max_terms=config.MAX_GLOSSARY_TERMS,
            boost=config.GLOSSARY_BOOST,
            boost_min=config.GLOSSARY_BOOST_MIN,
            boost_max=config.GLOSSARY_BOOST_MAX,
            accept=",".join(sorted(config.ALLOWED_EXTENSIONS)),
        )

    @app.post("/api/transcribe")
    def api_transcribe():
        upload = request.files.get("audio")
        if upload is None or not upload.filename:
            return error("Choose an audio file to upload.")
        ext = Path(upload.filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return error(f"Unsupported file type {ext or '(none)'}; use "
                         f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}.")
        try:
            terms = parse_glossary(request.form.get("glossary", ""))
        except GlossaryError as exc:
            return error(str(exc))
        try:
            boost = float(request.form.get("boost", config.GLOSSARY_BOOST))
        except ValueError:
            return error("Boost must be a number.")
        if not config.GLOSSARY_BOOST_MIN <= boost <= config.GLOSSARY_BOOST_MAX:
            return error(f"Boost must be between {config.GLOSSARY_BOOST_MIN} and "
                         f"{config.GLOSSARY_BOOST_MAX}.")

        job_id = uuid.uuid4().hex[:12]
        path = uploads / f"{job_id}{ext}"
        upload.save(path)
        try:
            wave = load_audio(path)
        except Exception as exc:  # unreadable or corrupt audio
            path.unlink(missing_ok=True)
            return error(f"Could not read the audio file ({exc}).")

        result = transcribe(wave, terms, model_loader(), boost)
        job = {"id": job_id, "filename": secure_filename(upload.filename) or "recording",
               "audio_path": str(path), "mimetype": upload.mimetype, **result}
        with lock:
            jobs[job_id] = job
        return jsonify(public(job))

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id):
        return jsonify(public(get_job(job_id)))

    @app.get("/api/jobs/<job_id>/audio")
    def api_audio(job_id):
        job = get_job(job_id)
        return send_file(job["audio_path"], mimetype=job["mimetype"], conditional=True)

    @app.post("/api/jobs/<job_id>/suggestions/<sid>")
    def api_review(job_id, sid):
        job = get_job(job_id)
        decision = (request.get_json(silent=True) or {}).get("decision")
        status = {"accept": "accepted", "reject": "rejected", "undo": "pending"}.get(decision)
        if status is None:
            return error('decision must be "accept", "reject" or "undo".')
        with lock:
            match = next((s for s in job["suggestions"] if s["id"] == sid), None)
            if match is None:
                abort(404, description="Unknown suggestion.")
            match["status"] = status
            job["final_text"] = final_text(job["segments"], job["suggestions"])
        return jsonify(public(job))

    @app.get("/api/jobs/<job_id>/export")
    def api_export(job_id):
        job = get_job(job_id)
        stem = Path(job["filename"]).stem or "transcript"
        if request.args.get("format") == "json":
            body = json.dumps(public(job), indent=2)
            return Response(body, mimetype="application/json", headers={
                "Content-Disposition": f'attachment; filename="{stem}.clearscribe.json"'})
        pending = sum(s["status"] == "pending" for s in job["suggestions"])
        note = (f"\n\n[{pending} glossary suggestion(s) not reviewed; standard wording kept]"
                if pending else "")
        return Response(job["final_text"] + note + "\n", mimetype="text/plain", headers={
            "Content-Disposition": f'attachment; filename="{stem}.txt"'})

    return app


def public(job) -> dict:
    return {k: v for k, v in job.items() if k not in ("audio_path", "mimetype")}


if __name__ == "__main__":
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"),
                     port=int(os.environ.get("PORT", 5000)), debug=False)
