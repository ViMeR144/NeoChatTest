#!/usr/bin/env python3
"""
API Server for Advanced Neural Network
- FastAPI-based REST API
- WebSocket support for streaming
- CUDA acceleration
- Multi-language support
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

# Import our neural network components
from main import NeuralNetwork, BPETokenizer, ModelConfig, GenerationConfig, TextGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
tokenizer = None
generator = None

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(100, ge=1, le=1000, description="Maximum number of tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Temperature for sampling")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(False, description="Whether to stream the response")

class GenerateResponse(BaseModel):
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    model_used: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Device used for inference")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field("1.0.0", description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device information")
    cuda_available: bool = Field(..., description="CUDA availability")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model, tokenizer, generator
    
    logger.info("Initializing Neural Network...")
    
    try:
        # Model configuration
        model_config = ModelConfig(
            vocab_size=4000,  # Updated to match BPE tokenizer
            n_embd=768,
            n_head=12,
            n_layer=12,
            n_positions=1024,
            dropout=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            tie_word_embeddings=True
        )
        
        # Initialize components
        tokenizer = BPETokenizer(model_config.vocab_size)
        model = NeuralNetwork(model_config)
        
        # Move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    finally:
        logger.info("Shutting down Neural Network...")

# Create FastAPI app
app = FastAPI(
    title="Advanced Neural Network API",
    description="API for transformer-based text generation with CUDA acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        model_loaded=model is not None,
        device=str(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"),
        cuda_available=torch.cuda.is_available()
    )

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create generation config
        generation_config = GenerationConfig(
            max_length=request.max_tokens + 100,  # Buffer for input tokens
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=True
        )
        
        # Create generator
        generator = TextGenerator(model, tokenizer, generation_config)
        
        # Generate text
        start_time = time.time()
        generated_text = generator.generate(request.prompt, request.max_tokens)
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Count tokens
        tokens_generated = len(tokenizer.encode(generated_text)) - len(tokenizer.encode(request.prompt))
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            generation_time_ms=generation_time_ms,
            model_used="transformer-neural-network",
            device=str(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.websocket("/api/v1/generate/stream")
async def generate_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming text generation"""
    await websocket.accept()
    
    if model is None or tokenizer is None:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close()
        return
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            # Parse request
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 100)
            temperature = data.get("temperature", 0.8)
            top_k = data.get("top_k", 50)
            top_p = data.get("top_p", 0.9)
            repetition_penalty = data.get("repetition_penalty", 1.1)
            
            if not prompt:
                await websocket.send_json({"error": "Prompt is required"})
                continue
            
            # Create generation config
            generation_config = GenerationConfig(
                max_length=max_tokens + 100,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True
            )
            
            # Create generator
            generator = TextGenerator(model, tokenizer, generation_config)
            
            # Stream generation
            try:
                for partial_text in generator.generate_streaming(prompt, max_tokens):
                    await websocket.send_json({
                        "text": partial_text,
                        "streaming": True
                    })
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
                
                # Send final message
                await websocket.send_json({
                    "text": "",
                    "streaming": False,
                    "completed": True
                })
                
            except Exception as e:
                await websocket.send_json({"error": f"Generation failed: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/api/v1/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "transformer",
        "vocab_size": model.config.vocab_size,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "n_layer": model.config.n_layer,
        "n_positions": model.config.n_positions,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(next(model.parameters()).device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/api/v1/tokenizer/info")
async def tokenizer_info():
    """Get tokenizer information"""
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    return {
        "type": "BPE",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_tokens,
        "next_token_id": tokenizer.next_token_id
    }

@app.post("/api/v1/tokenizer/encode")
async def encode_text(text: str):
    """Encode text to tokens"""
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        tokens = tokenizer.encode(text)
        return {"tokens": tokens, "count": len(tokens)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {str(e)}")

@app.post("/api/v1/tokenizer/decode")
async def decode_tokens(tokens: List[int]):
    """Decode tokens to text"""
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        text = tokenizer.decode(tokens)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1,  # Single worker for model sharing
        log_level="info"
    )
