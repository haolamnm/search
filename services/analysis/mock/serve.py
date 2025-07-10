import argparse

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class MockQueryRequest(BaseModel):
    query: str


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="mock analysis service")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.post("/encode")
    def encode(request: MockQueryRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Mock encoding: return a random float32 vector
        rng = np.random.default_rng(seed=args.seed)
        feature_vector = rng.random(args.length).astype(np.float32).tolist()
        return {"feature_vector": feature_vector, "length": len(feature_vector)}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mock extractor")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to run the server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host to run the server on",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=512,
        choices=[384, 512, 768],
        help="length of the feature vector to return",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
