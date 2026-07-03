"""FastAPI web app: upload a forehand clip, get back annotated video + coaching feedback."""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .pipeline import analyze_forehand, LowPoseConfidenceError
from .feedback import OllamaUnavailableError

REFERENCE_MODEL_PATH = "models/reference_model.pkl"

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = str(Path(tmp_dir) / video.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        output_dir = str(Path(tmp_dir) / "output")

        try:
            result = analyze_forehand(video_path, REFERENCE_MODEL_PATH, output_dir)
        except (LowPoseConfidenceError, OllamaUnavailableError) as exc:
            return templates.TemplateResponse(
                "upload.html", {"request": request, "error": str(exc)}
            )

        served_dir = Path("static/results")
        served_dir.mkdir(parents=True, exist_ok=True)

        annotated_name = "annotated.mp4"
        shutil.copy(result.annotated_video_path, served_dir / annotated_name)

        phase_urls = {}
        for phase_name, path in result.phase_frame_paths.items():
            dest_name = f"{phase_name}.jpg"
            shutil.copy(path, served_dir / dest_name)
            phase_urls[phase_name] = f"/static/results/{dest_name}"

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "feedback_text": result.feedback_text,
                "annotated_video_url": f"/static/results/{annotated_name}",
                "phase_urls": phase_urls,
            },
        )


app.mount("/static", StaticFiles(directory="static"), name="static")
