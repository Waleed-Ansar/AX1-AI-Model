from fastapi import APIRouter, FastAPI
from api import API
import uvicorn


app = FastAPI()
router = APIRouter()
api = API()

router.include_router(api.router)

app.include_router(router, prefix='/get link')

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)