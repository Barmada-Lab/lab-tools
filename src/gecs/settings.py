from pydantic import BaseSettings
from pathlib import Path

USER_CONFIG_HOME = Path.home() / ".config" / "gecs"


class Settings(BaseSettings):
    app_name: str = "GECS"

    log_level: str = "DEBUG"

    cvat_url: str = "http://localhost:8080"
    cvat_username: str = ""
    cvat_password: str = ""
    cvat_org_slug: str = "barma"

    class Config:
        env_file = USER_CONFIG_HOME / "gecs.env"
        env_file_encoding = "utf-8"


settings = Settings()