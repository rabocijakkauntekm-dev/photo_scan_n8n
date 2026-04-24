import json

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from scanner import process_document

app = FastAPI(title="Document Scan API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/enhance")
async def enhance_photo(
    file: UploadFile = File(...),
    response_format: str = Query(default="image", pattern="^(image|json)$"),
    scan_mode: str = Query(default="color", pattern="^(color|clean_gray|bw)$"),
) -> Response:
    """
    Accepts an image and returns enhanced output.

    response_format=image -> binary PNG output
    response_format=json  -> metadata JSON only
    scan_mode=color       -> color enhancement (default)
    scan_mode=clean_gray  -> natural grayscale enhancement
    scan_mode=bw          -> high-contrast black/white
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        scan, meta = process_document(payload, scan_mode=scan_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    ok, encoded = cv2.imencode(".png", scan)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")

    if response_format == "json":
        return JSONResponse({"ok": True, "meta": meta})

    headers = {
        "X-Scan-Meta": json.dumps(meta, ensure_ascii=True),
        "Content-Disposition": f'inline; filename="enhanced_{file.filename or "image"}.png"',
    }
    return Response(content=encoded.tobytes(), media_type="image/png", headers=headers)


@app.post("/scan")
async def backward_compatible_scan(
    file: UploadFile = File(...),
    response_format: str = Query(default="image", pattern="^(image|json)$"),
    scan_mode: str = Query(default="color", pattern="^(color|clean_gray|bw)$"),
) -> Response:
    """Backward-compatible alias for old clients."""
    return await enhance_photo(file=file, response_format=response_format, scan_mode=scan_mode)
