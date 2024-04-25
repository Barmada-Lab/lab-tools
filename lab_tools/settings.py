from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path

USER_CONFIG_HOME = Path.home() / ".config"


class Settings(BaseSettings):
    app_name: str = "lab_tools"

    log_level: str = "INFO"

    #  CVAT config
    cvat_url: str = ""
    cvat_username: str = ""
    cvat_password: str = ""
    cvat_org_slug: str = "barma"

    #  Path to FiftyOne dataset storage -- stores png-converted images
    fo_cache: Path = Path("/data/.focache/")

    #  Path to shared model storage -- stores serialized models
    models_path: Path = Path("/nfs/turbo/shared/models")

    #  Path to shared collection storage -- stores annoted datasets for training
    collections_path: Path = Path("/nfs/turbo/shared/collections")

    celery_broker_url: str = "redis://localhost:6379"

    @field_validator("models_path", "collections_path")
    @classmethod
    def exists(cls, path: Path):
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        return path

    class Config:
        env_file = USER_CONFIG_HOME / "lab_tools.env"
        env_file_encoding = "utf-8"
        env_prefix = 'lab_tools_'


settings = Settings()
