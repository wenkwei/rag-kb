from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.auth import router as auth_router, verify_token
from api.documents import router as documents_router
from api.chat import router as chat_router
from api.config import router as config_router
from api.users import router as users_router
from api.badcase import router as badcase_router
from api.conversations import router as conversations_router
from api.evaluation import router as evaluation_router
from api.audit import router as audit_router

app = FastAPI(title="RAG 知识库问答系统", version="1.0.0")

# CORS — allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public routes
app.include_router(auth_router)

# Protected routes (require valid token)
app.include_router(documents_router, dependencies=[Depends(verify_token)])
app.include_router(chat_router, dependencies=[Depends(verify_token)])
app.include_router(config_router, dependencies=[Depends(verify_token)])
app.include_router(users_router, dependencies=[Depends(verify_token)])
app.include_router(badcase_router, dependencies=[Depends(verify_token)])
app.include_router(conversations_router, dependencies=[Depends(verify_token)])
app.include_router(evaluation_router, dependencies=[Depends(verify_token)])
app.include_router(audit_router, dependencies=[Depends(verify_token)])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag-knowledge-base"}
