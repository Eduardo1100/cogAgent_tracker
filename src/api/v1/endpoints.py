import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.storage import cache, s3
from src.storage.database import get_db
from src.storage.models import EpisodeRun, ExperimentRun, Prediction

router = APIRouter()


@router.post("/test-stack")
async def run_integration_test(db: Session = Depends(get_db)):
    # 1. Save to Postgres
    new_pred = Prediction(filename="test.jpg", result="AI_SUCCESS", confidence=99)
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)

    # 2. Save to Redis (Cache the ID for 60 seconds)
    cache.set_cache("last_id", str(new_pred.id), expire=60)

    # 3. Save a "dummy log" to MinIO
    s3_uri = s3.upload_file(b"Test Log Data", "ai-logs", f"log_{new_pred.id}.txt")

    return {
        "database_id": new_pred.id,
        "cached_val": cache.get_cache("last_id"),
        "storage_uri": s3_uri,
        "status": "Integration Success",
    }


@router.get("/experiments/{experiment_id}/agents-config")
async def get_agents_config(experiment_id: int, db: Session = Depends(get_db)):
    cache_key = f"agents_config:{experiment_id}"
    cached = cache.get_cache(cache_key)
    if cached:
        return {"source": "cache", "agents_config": json.loads(cached)}

    experiment = (
        db.query(ExperimentRun).filter(ExperimentRun.id == experiment_id).first()
    )
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.agents_config is None:
        raise HTTPException(
            status_code=404, detail="No agents_config recorded for this experiment"
        )

    cache.set_cache(cache_key, json.dumps(experiment.agents_config), expire=86400)
    return {"source": "db", "agents_config": experiment.agents_config}


@router.get("/experiments/")
async def list_experiments(db: Session = Depends(get_db)):
    experiments = db.query(ExperimentRun).order_by(ExperimentRun.id.desc()).all()
    return experiments


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = (
        db.query(ExperimentRun).filter(ExperimentRun.id == experiment_id).first()
    )
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.get("/experiments/{experiment_id}/episodes")
async def list_episodes(experiment_id: int, db: Session = Depends(get_db)):
    experiment = (
        db.query(ExperimentRun).filter(ExperimentRun.id == experiment_id).first()
    )
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    episodes = (
        db.query(EpisodeRun)
        .filter(EpisodeRun.experiment_id == experiment_id)
        .order_by(EpisodeRun.game_number)
        .all()
    )
    return episodes


@router.get("/episodes/{episode_id}")
async def get_episode(episode_id: int, db: Session = Depends(get_db)):
    episode = db.query(EpisodeRun).filter(EpisodeRun.id == episode_id).first()
    if not episode:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


@router.put("/experiments/{experiment_id}/agents-config")
async def put_agents_config(
    experiment_id: int,
    payload: dict,
    db: Session = Depends(get_db),
):
    experiment = (
        db.query(ExperimentRun).filter(ExperimentRun.id == experiment_id).first()
    )
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    experiment.agents_config = payload["agents_config"]
    db.commit()

    cache_key = f"agents_config:{experiment_id}"
    cache.set_cache(cache_key, json.dumps(payload["agents_config"]), expire=86400)
    return {"status": "ok", "experiment_id": experiment_id}
