"""client.py — Type-safe HTTP client for the Data Preparation Pipeline environment."""

from __future__ import annotations
import requests
from models import Action, ResetResponse, StepResponse, EnvironmentState


class DataPrepClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = requests.Session()

    def reset(self, task_id: int = 1) -> ResetResponse:
        r = self._session.post(f"{self.base_url}/reset",
                               json={"task_id": task_id}, timeout=self.timeout)
        r.raise_for_status()
        return ResetResponse(**r.json())

    def step(self, action: Action) -> StepResponse:
        r = self._session.post(f"{self.base_url}/step",
                               json=action.dict(exclude_none=True), timeout=self.timeout)
        r.raise_for_status()
        return StepResponse(**r.json())

    def state(self) -> EnvironmentState:
        r = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return EnvironmentState(**r.json())

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False