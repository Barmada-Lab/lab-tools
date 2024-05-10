import logging

import dask.config
from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path
import dask


USER_CONFIG_PATH = Path.home() / ".config" / "cytomancer.env"


class CytomancerConfig(BaseSettings):

    log_level: str = "INFO"

    #  CVAT config
    cvat_url: str = ""
    cvat_username: str = ""
    cvat_password: str = ""
    cvat_org: str = "barma"

    #  Path to FiftyOne dataset storage -- stores png-converted images
    fo_cache: Path = Path("/data/.focache/")

    #  Path to shared model storage -- stores serialized models
    models_dir: Path = Path("/nfs/turbo/shared/models")

    #  Path to shared collection storage -- stores annoted datasets for training
    collections_dir: Path = Path("/nfs/turbo/shared/collections")

    #  Url of celery broker; probably a redis instance
    celery_broker_url: str = "redis://localhost:6379"

    @field_validator("models_dir", "collections_dir")
    @classmethod
    def exists(cls, path: Path):
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        return path

    def save(self):
        with open(USER_CONFIG_PATH, "w") as f:
            for k, v in self.model_dump().items():
                f.write(f"{k}={v}\n")

    class Config:
        env_file = USER_CONFIG_PATH
        env_file_encoding = "utf-8"
        env_prefix = 'cytomancer_'
        extra = "ignore"


config = CytomancerConfig()
logging.basicConfig(level=config.log_level)
dask.config.set({"distributed.scheduler.worker-ttl": "5m"})
