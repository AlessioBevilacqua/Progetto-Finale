from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from travel_agent_api.routes import chat_router

app = FastAPI()

#configurazione CORS
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      
    allow_credentials=True,     
    allow_methods=["*"],        
    allow_headers=["*"],        
)

app.include_router(
    chat_router.router,   
    tags=["Chat"],        
    prefix="/chat"        
)