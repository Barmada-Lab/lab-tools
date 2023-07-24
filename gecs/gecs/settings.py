from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "GECS"

    cvat_url: str = "http://localhost:8080"
    cvat_username: str = ""
    cvat_password: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()