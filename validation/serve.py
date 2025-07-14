import uvicorn
import argparse
import time
import torch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from models import ValidateRequest, ValidateResponse
from validation_endpoint import Validation

app = FastAPI()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    return args

args = get_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.validation = Validation()
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        
app.router.lifespan_context = lifespan

@app.post("/validate")
async def validate(data: ValidateRequest) -> ValidateResponse:
    start = time.time()
    try:
        score = app.state.validation.validate(data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        torch.cuda.empty_cache()
        
    print(f"Total time: {time.time() - start}")
    return score

async def serve_validate_m_test(obj_file_path: str, preview_file_path: str, prompt: str):
    try:
        validation = Validation()
        score = validation.validate_m_test(obj_file_path, preview_file_path, prompt)        
    except Exception as e:
        print(f"{e}")
    finally:
        torch.cuda.empty_cache()
    return score

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
