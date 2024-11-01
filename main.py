from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Haystack import HaystackRouter

app = FastAPI()

#add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to match the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():    
    return [{'id': 1,"message": "Hello a Trong"}, {'id': 2, "message": "Hello a Trung"}]

app.include_router(HaystackRouter, prefix="/haystack", tags=["haystack"])