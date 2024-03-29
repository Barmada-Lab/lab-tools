from pydantic_settings import BaseSettings
from pathlib import Path

USER_CONFIG_HOME = Path.home() / ".config" / "gecs"


class Settings(BaseSettings):
    app_name: str = "GECS"

    log_level: str = "INFO"

    cvat_url: str = ""
    cvat_username: str = ""
    cvat_password: str = ""
    cvat_org_slug: str = "barma"

    fo_cache: Path = Path("/data/.focache/")

    class Config:
        env_file = USER_CONFIG_HOME / "gecs.env"
        env_file_encoding = "utf-8"


settings = Settings()
