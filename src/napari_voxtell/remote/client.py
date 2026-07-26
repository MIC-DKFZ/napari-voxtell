"""Torch-free HTTP client for a remote ``voxtell-server``.

Only depends on numpy + httpx + blosc2, so a laptop can drive inference running on a
workstation/cluster without installing torch, nnU-Net or the voxtell package. Mirrors
the local ``VoxTellPredictor`` usage closely enough that the napari widget can swap
between them.

The client/server design is inspired by MIC-DKFZ's napari-nninteractive / nnInteractive
(Apache-2.0).
"""

from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional, Tuple

import httpx
import numpy as np

from napari_voxtell.remote.serialization import pack_array, unpack_array

META_HEADER = "X-Meta"
CONTENT_TYPE_OCTET_STREAM = "application/octet-stream"


class VoxTellRemoteClient:
    """Thin wrapper around the ``voxtell-server`` HTTP API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 600.0):
        # Short connect timeout (fail fast on a bad URL), long read timeout (uploads and
        # result downloads of full volumes are large); the event stream overrides this.
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=self._auth_headers(api_key),
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    @staticmethod
    def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def healthz(self) -> bool:
        """Return True if the server answers its health check."""
        resp = self._client.get("/healthz")
        resp.raise_for_status()
        return bool(resp.json().get("ok"))

    def capabilities(self) -> dict:
        """Model name, patch size and the list of precomputed prompt embeddings."""
        resp = self._client.get("/capabilities")
        resp.raise_for_status()
        return resp.json()

    def upload_image(
        self, nifti_bytes: bytes, include_image: bool = True
    ) -> Tuple[str, Optional[np.ndarray], tuple, tuple]:
        """Upload a .nii.gz; return (image_id, array_or_None, spacing, shape).

        With ``include_image`` false the server skips returning the reoriented array
        (``array`` is None) - used when the GUI already loaded the image locally and
        only needs the ``image_id`` and a shape to sanity-check.
        """
        resp = self._client.post(
            "/images",
            content=nifti_bytes,
            params={"include_image": str(include_image).lower()},
            headers={"Content-Type": CONTENT_TYPE_OCTET_STREAM},
        )
        resp.raise_for_status()
        meta = json.loads(resp.headers[META_HEADER])
        array = unpack_array(resp.content) if include_image and resp.content else None
        return meta["image_id"], array, tuple(meta["spacing"]), tuple(meta["shape"])

    def start_job(self, image_id: str, prompts: List[str], keep_largest: bool = False) -> str:
        """Start a background segmentation; return its job_id."""
        resp = self._client.post(
            "/jobs",
            json={"image_id": image_id, "prompts": prompts, "keep_largest": keep_largest},
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def stream_events(self, job_id: str) -> Iterator[dict]:
        """Yield NDJSON progress dicts until a terminal ``{"status": ...}`` event.

        Progress events are ``{"done", "total"}``; OOM notices are
        ``{"event": "oom_fallback", "message"}``; the last event carries ``status``
        of ``done`` / ``cancelled`` / ``error``.
        """
        with self._client.stream("GET", f"/jobs/{job_id}/events", timeout=None) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def get_result(self, job_id: str) -> np.ndarray:
        """Download the finished masks, shape ``(num_prompts, Z, Y, X)`` uint8."""
        resp = self._client.get(f"/jobs/{job_id}/result")
        resp.raise_for_status()
        return unpack_array(resp.content)

    def cancel(self, job_id: str) -> None:
        """Request cooperative cancellation of a running job."""
        self._client.post(f"/jobs/{job_id}/cancel")

    def export(self, image_id: str, labelmap: np.ndarray) -> bytes:
        """Write ``labelmap`` back in the original orientation; return .nii.gz bytes."""
        resp = self._client.post(
            f"/images/{image_id}/export",
            content=pack_array(labelmap),
            headers={"Content-Type": CONTENT_TYPE_OCTET_STREAM},
        )
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self._client.close()
