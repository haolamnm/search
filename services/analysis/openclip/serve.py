import argparse

import open_clip
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class OpenCLIPQueryEncoder:
    def __init__(self, model_name: str, pretrained: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.context_length = self.model.context_length
        self.model = torch.compile(self.model)

    def encode(self, query: str) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                context_length=self.context_length,
            ).to(self.device)
            features = self.model.encode_text(inputs).float()  # type: ignore
            features = F.normalize(features, dim=-1, p=2)

            return features.cpu().squeeze().tolist()


class QueryRequest(BaseModel):
    query: str


def create_app(model_name: str, pretrained: str, title: str) -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    encoder = OpenCLIPQueryEncoder(model_name, pretrained)
    app.state.encoder = encoder

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.post("/encode")
    def encode(request: QueryRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        try:
            feature = encoder.encode(query)
            return {"feature": feature, "length": len(feature)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openclip encoder")
    parser.add_argument(
        "--pretrained",
        default="laion2b_s32b_b82k",
        type=str,
        choices=[
            "laion2b_s32b_b82k",
            "datacomp_xl_s13b_b90k",
        ],
        help="pretrained model to use for feature extraction",
    )
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
    args = parser.parse_args()

    model_name = "ViT-L-14"
    if args.pretrained == "laion2b_s32b_b82k":
        title = "clip-laion encoder"
    else:
        title = "clip-datacomp encoder"
    app = create_app("ViT-L-14", args.pretrained, title)
    uvicorn.run(app, host=args.host, port=args.port)
