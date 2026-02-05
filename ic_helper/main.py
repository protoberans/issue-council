from dotenv import load_dotenv
load_dotenv()
import uvicorn

from .api import create_app
from .config import LISTEN_HOST, LISTEN_PORT

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
