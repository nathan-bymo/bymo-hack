from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

# before curl -fsS https://www.kaggle.com/.well-known/oauth-authorization-server

data_dir = Path(__file__).resolve().parents[2] / "data"
data_dir.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()
api.competition_download_files(
    "hackapertus-galaxus", path=str(data_dir), quiet=False, unzip=True
)

print("Path to competition files:", data_dir)
