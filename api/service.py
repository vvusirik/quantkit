from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic.dataclasses import dataclass
from typing import Optional
import datetime as dt

app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Allows auth headers, cookies, etc..
    allow_credentials=True,
    # GET, PUT, POST, etc..
    allow_methods=["*"],
    # HTTP Headers
    allow_headers=["*"],
)


@dataclass
class Item:
    name: str
    price: float
    stars: int
    list_date: dt.date
    is_offer: Optional[bool] = None


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str]):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item_name": item.name}
