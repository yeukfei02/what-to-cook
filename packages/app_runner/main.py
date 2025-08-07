from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bacinet import BacinetMiddleware
from routes.routes import data_router


app = FastAPI()

# middleware

# cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# helmet
app.add_middleware(BacinetMiddleware)

# routes
app.include_router(data_router)
