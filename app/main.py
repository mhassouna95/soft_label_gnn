"""InteractiveAI recommendation API for the SoftGNN agent.

InteractiveAI posts the grid context of a critical event. The API restores the
Grid2Op observation it contains, lets the agent propose up to
``N_RECOMMENDATIONS`` actions, and returns them in InteractiveAI's
recommendation format.

Run from the repository root:

    API_TOKEN=<token> GRID2OP_ENV=<path to ai4realnet_small> \\
        uvicorn app.main:app --host 0.0.0.0 --port 8000

Request recommendations:

    curl -X POST http://localhost:8000/api/v1/recommendation \\
        -H "Content-Type: application/json" \\
        -H "Authorization: Bearer $API_TOKEN" \\
        --data @app/sample_request.json
"""
import logging
import os
import pickle
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import grid2op
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from grid2op.Observation import CompleteObservation
from lightsim2grid import LightSimBackend
from pydantic import BaseModel, Field

from GNNAgent import GNNAgent
from app.formatting import format_recommendation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("softgnn-agent-api")


@dataclass(frozen=True)
class Settings:
    """Service configuration, read from environment variables."""

    grid2op_env: str
    model_path: Path
    actions_path: Path
    scaler_path: Path
    n_recommendations: int
    extra_simulation_budget: int
    best_action_threshold: float
    max_action_sim: int
    agent_type: int

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            grid2op_env=env("GRID2OP_ENV", "ai4realnet_small"),
            model_path=Path(env("MODEL_PATH", PROJECT_ROOT / "data" / "best_model")),
            actions_path=Path(env("ACTIONS_PATH", PROJECT_ROOT / "data" / "actions" / "soft_actions.npy")),
            scaler_path=Path(env("SCALER_PATH", PROJECT_ROOT / "data" / "scaler_all.pkl")),
            n_recommendations=int(env("N_RECOMMENDATIONS", 3)),
            extra_simulation_budget=int(env("EXTRA_SIMULATION_BUDGET", 300)),
            best_action_threshold=float(env("BEST_ACTION_THRESHOLD", 0.95)),
            max_action_sim=int(env("MAX_ACTION_SIM", 2000)),
            agent_type=int(env("AGENT_TYPE", 2)),
        )


class InvalidObservation(ValueError):
    """The request's observation does not belong to the grid the agent runs on."""


class AgentService:
    """Holds the Grid2Op environment and the agent and answers requests.

    The agent and the observation object are reused across requests and are
    not safe to use from several threads at once, so requests are handled one
    at a time.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        logger.info("Loading Grid2Op environment %s", settings.grid2op_env)
        self.env = grid2op.make(settings.grid2op_env, backend=LightSimBackend(),
                                observation_class=CompleteObservation)

        with open(settings.scaler_path, "rb") as fp:
            scaler = pickle.load(fp)

        logger.info("Loading agent from %s", settings.model_path)
        self.agent = GNNAgent(
            action_space=self.env.action_space,
            model_path=settings.model_path,
            action_space_file=settings.actions_path,
            best_action_threshold=settings.best_action_threshold,
            scaler=scaler,
            topo=True,
            max_action_sim=settings.max_action_sim,
        )

        self.observation = self.env.reset()
        self._expected_sizes = {"rho": self.env.n_line, "topo_vect": self.env.dim_topo,
                                "load_p": self.env.n_load, "gen_p": self.env.n_gen}
        self._lock = threading.Lock()
        logger.info("Agent ready")

    def recommend(self, observation_json: dict) -> List[dict]:
        """Propose recommendations for an observation serialized by Grid2Op's to_json."""
        for key, size in self._expected_sizes.items():
            values = observation_json.get(key)
            if not isinstance(values, list) or len(values) != size:
                raise InvalidObservation(
                    f"context.observation['{key}'] must be a list of {size} values for this grid")

        with self._lock:
            try:
                self.observation.from_json(observation_json)
            except Exception as e:
                raise InvalidObservation(f"context.observation could not be restored: {e}") from e

            start = time.perf_counter()
            proposals = self.agent.recommend(
                self.observation,
                n_recommendations=self.settings.n_recommendations,
                extra_simulation_budget=self.settings.extra_simulation_budget,
            )
            recommendations = [
                format_recommendation(proposal["action"], self.observation, self.settings.agent_type)
                for proposal in proposals
            ]

        logger.info("rho.max %.4f -> %d recommendation(s) in %.2fs: %s",
                    self.observation.rho.max(), len(recommendations), time.perf_counter() - start,
                    ", ".join(f"action {p['action_id']} (rank {p['rank']}, rho {p['simulated_rho']:.4f})"
                              for p in proposals))
        return recommendations


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = AgentService(Settings.from_env())
    yield


app = FastAPI(title="SoftGNN agent API", lifespan=lifespan)

_bearer = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> None:
    """Accept only requests carrying the bearer token set in API_TOKEN."""
    expected = os.environ.get("API_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_TOKEN environment variable is not set",
        )
    if not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RecommendationRequest(BaseModel):
    """Payload InteractiveAI sends when an operator asks for recommendations."""

    event: dict = Field(default_factory=dict)
    context: dict
    cognitive_snapshot: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/recommendation", dependencies=[Depends(verify_token)])
def get_recommendation(payload: RecommendationRequest, request: Request) -> List[dict]:
    observation = payload.context.get("observation")
    if not isinstance(observation, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="context.observation must be a Grid2Op observation serialized as JSON",
        )

    logger.info("Recommendation requested for event %s", payload.event.get("event_type", "<unspecified>"))
    try:
        return request.app.state.service.recommend(observation)
    except InvalidObservation as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
