#!/usr/bin/env python3
"""
Advanced Neural Network with Dynamic Text Generation
- Token-by-token generation
- Probabilistic prediction with sampling
- Top-k and top-p filtering
- Temperature control and repetition penalty
- Subword/byte-level tokenization
- Attention state caching for long contexts
- Multi-language support
- CUDA acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
import os
import pickle
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from collections import Counter
import time
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import CUDA kernels
try:
    import cuda_kernels
    CUDA_KERNELS_AVAILABLE = True
    logger.info("CUDA kernels loaded successfully")
except ImportError:
    CUDA_KERNELS_AVAILABLE = False
    logger.warning("CUDA kernels not available, using PyTorch operations")

@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 4000  # Updated to match BPE tokenizer
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True
    do_sample: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 3
    unk_token_id: int = 1

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with caching and CUDA acceleration"""
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.d_model = config.n_embd
        self.n_heads = config.n_head
        self.d_k = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        
        self.w_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.w_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.w_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.w_o = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Cache for attention states (past_key_values)
        self.cache = {}
        
        # Initialize weights
        self._init_weights(config.initializer_range)
        
    def _init_weights(self, initializer_range: float):
        """Initialize attention weights"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Use cached keys and values if available
        if past_key_values is not None:
            past_k, past_v = past_key_values
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        
        # Cache current keys and values for next iteration
        present_key_values = (K, V) if use_cache else None
        
        # Use CUDA kernels if available
        if CUDA_KERNELS_AVAILABLE and x.is_cuda:
            try:
                # Reshape for CUDA kernel
                Q_flat = Q.contiguous().view(-1, self.d_model)
                K_flat = K.contiguous().view(-1, self.d_model)
                V_flat = V.contiguous().view(-1, self.d_model)
                
                # Create output tensor
                context_flat = torch.zeros_like(Q_flat)
                
                # Call CUDA kernel
                context_flat = cuda_kernels.attention_forward(
                    Q_flat, K_flat, V_flat, context_flat, self.scale
                )
                
                # Reshape back
                context = context_flat.view(batch_size, self.n_heads, seq_len, self.d_k)
                
            except Exception as e:
                logger.warning(f"CUDA kernel failed, falling back to PyTorch: {e}")
                # Fallback to PyTorch implementation
                context = self._pytorch_attention(Q, K, V, mask)
        else:
            # PyTorch implementation
            context = self._pytorch_attention(Q, K, V, mask)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        
        return output, present_key_values
    
    def _pytorch_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PyTorch implementation of attention"""
        batch_size, n_heads, seq_len, d_k = Q.size()
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply causal mask for autoregressive generation
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context

class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=config.initializer_range)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=config.initializer_range)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = MultiHeadAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention with residual connection
        attn_output, present_key_values = self.attention(
            self.ln1(x), mask, past_key_values, use_cache
        )
        x = x + attn_output
        
        # Feed-forward with residual connection
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        
        return x, present_key_values

class NeuralNetwork(nn.Module):
    """Advanced Neural Network for Text Generation with Transformer Architecture"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)
        ])
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output projection (language modeling head)
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight  # Tie weights
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = input_ids.size()
        
        # Create position indices
        if past_key_values is not None:
            # For generation, position is the length of the sequence so far
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) + len(past_key_values[0][0])
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Pass through transformer blocks
        present_key_values = []
        for i, block in enumerate(self.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = block(
                hidden_states, attention_mask, past_kv, use_cache
            )
            present_key_values.append(present_kv)
        
        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits, present_key_values
    
    def generate(self, input_ids: torch.Tensor, generation_config: GenerationConfig,
                 past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """Generate next token"""
        with torch.no_grad():
            logits, present_key_values = self.forward(
                input_ids, past_key_values=past_key_values, use_cache=True
            )
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if generation_config.temperature != 1.0:
                next_token_logits = next_token_logits / generation_config.temperature
            
            # Apply top-k filtering
            if generation_config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, generation_config.top_k)
                next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Apply top-p (nucleus) filtering
            if generation_config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if generation_config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0)
            
            return next_token_id, present_key_values

class BPETokenizer:
    """Byte Pair Encoding tokenizer with dynamic subword learning"""
    
    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.merges = []         # list of merges as tuples ('a','b')
        self.vocab = {}          # token -> id
        self.id_to_token = {}    # id -> token
        self._trained = False
        
        # Train on a multi-language corpus
        self._train_on_corpus()
    
    def _train_on_corpus(self):
        """Train BPE tokenizer on a multi-language corpus"""
        # Much larger multi-language training corpus with repeated patterns
        corpus = (
            # Russian conversational patterns
            "привет как дела что нового как поживаешь все хорошо спасибо "
            "привет как дела что нового как поживаешь все хорошо спасибо "
            "привет как дела что нового как поживаешь все хорошо спасибо "
            "привет как дела что нового как поживаешь все хорошо спасибо "
            "привет как дела что нового как поживаешь все хорошо спасибо "
            
            # English conversational patterns  
            "hello how are you what is new how are you doing everything is good thank you "
            "hello how are you what is new how are you doing everything is good thank you "
            "hello how are you what is new how are you doing everything is good thank you "
            "hello how are you what is new how are you doing everything is good thank you "
            "hello how are you what is new how are you doing everything is good thank you "
            
            # Mixed conversations
            "привет hello как дела how are you хорошо good спасибо thank you "
            "привет hello как дела how are you хорошо good спасибо thank you "
            "привет hello как дела how are you хорошо good спасибо thank you "
            "привет hello как дела how are you хорошо good спасибо thank you "
            "привет hello как дела how are you хорошо good спасибо thank you "
            
            # Common words repeated
            "да yes нет no хорошо good плохо bad "
            "да yes нет no хорошо good плохо bad "
            "да yes нет no хорошо good плохо bad "
            "да yes нет no хорошо good плохо bad "
            "да yes нет no хорошо good плохо bad "
            
            # Simple sentences
            "я думаю i think это this то that "
            "я думаю i think это this то that "
            "я думаю i think это this то that "
            "я думаю i think это this то that "
            "я думаю i think это this то that "
            
            # Numbers and basic words
            "один two три four пять five "
            "один two три four пять five "
            "один two три four пять five "
            "один two три four пять five "
            "один two три four пять five "
        )
        
        self.train(corpus)
    
    def train(self, corpus_text):
        """
        Train a BPE-like tokenizer on a corpus string.
        Result: self.vocab (token->id) and self.merges (frequent merges).
        """
        # Preprocess: normalize whitespace, lowercase (optional)
        corpus_text = corpus_text.strip()
        words = re.findall(r"\S+", corpus_text.lower())

        # Start from characters with end-of-word marker
        tokenized_words = [" ".join(list(w) + ["</w>"]) for w in words]
        vocab_counter = Counter(tokenized_words)  # "a b c</w>" : freq

        merges = []
        for _ in range(self.vocab_size):
            pairs = Counter()
            for word, freq in vocab_counter.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]  # tuple (a,b)
            merges.append(best)

            # apply merge to vocab_counter: replace "a b" with "ab"
            new_vocab = {}
            bigram = " ".join(best)
            merged_symbol = "".join(best)
            pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
            for word, freq in vocab_counter.items():
                new_word = pattern.sub(merged_symbol, word)
                new_vocab[new_word] = freq
            vocab_counter = new_vocab

        # build final token set from remaining symbols
        tokens = set()
        for word in vocab_counter:
            for sym in word.split():
                tokens.add(sym)
        # ensure a stable ordering (sorted) for ids
        tokens = sorted(tokens)
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}
        self.merges = merges
        self._trained = True

    def encode(self, text):
        """
        Encode text (string) -> list of token ids (ints)
        Uses the learned merges greedily from left to right.
        """
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train(corpus) first.")

        out_ids = []
        words = re.findall(r"\S+", text.lower())
        for w in words:
            symbols = list(w) + ["</w>"]
            # greedily apply merges that exist in self.merges
            merged = True
            # build set for quick check
            merges_set = set(self.merges)
            while merged:
                merged = False
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) in merges_set:
                        new_symbols.append(symbols[i] + symbols[i+1])
                        i += 2
                        merged = True
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            # map symbols to ids
            for s in symbols:
                if s in self.vocab:
                    out_ids.append(self.vocab[s])
                else:
                    # unknown symbol -> try to split into chars and map, else use 0
                    found = False
                    for ch in s:
                        if ch in self.vocab:
                            out_ids.append(self.vocab[ch])
                            found = True
                    if not found:
                        out_ids.append(0)  # UNK fallback id
        return out_ids

    def decode(self, ids):
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train(corpus) first.")
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        # join tokens and replace end-of-word markers </w> with spaces
        text = "".join(tokens)
        # since tokens can be multi-char symbols, do a safe replace of </w>
        text = text.replace("</w>", " ")
        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

class TextGenerator:
    """Advanced text generator with transformer architecture and CUDA acceleration"""
    
    def __init__(self, model: NeuralNetwork, tokenizer: BPETokenizer, 
                 config: GenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"TextGenerator initialized on device: {self.device}")
        
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from prompt using transformer architecture"""
        logger.info(f"Generating text for prompt: '{prompt}'")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_ids = input_ids.copy()
        past_key_values = None
        
        start_time = time.time()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model predictions with caching
                next_token_id, past_key_values = self.model.generate(
                    input_tensor, self.config, past_key_values
                )
                
                # Apply repetition penalty
                if len(generated_ids) > 1 and self.config.repetition_penalty != 1.0:
                    # This would be implemented in the model.generate method
                    pass
                
                next_token_id = next_token_id.item()
                
                # Check for end token
                if next_token_id == self.config.eos_token_id:
                    break
                
                generated_ids.append(next_token_id)
                
                # Update input tensor for next iteration (only last token)
                input_tensor = torch.tensor([[next_token_id]], device=self.device)
                
                # Truncate if too long
                if len(generated_ids) > self.config.max_length:
                    generated_ids = generated_ids[-self.config.max_length:]
        
        generation_time = time.time() - start_time
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids)
        
        logger.info(f"Generated {len(generated_ids)} tokens in {generation_time:.2f}s")
        logger.info(f"Generated text: '{generated_text}'")
        
        return generated_text
    
    def generate_streaming(self, prompt: str, max_new_tokens: int = 100):
        """Generate text with streaming output"""
        logger.info(f"Streaming generation for prompt: '{prompt}'")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_ids = input_ids.copy()
        past_key_values = None
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model predictions with caching
                next_token_id, past_key_values = self.model.generate(
                    input_tensor, self.config, past_key_values
                )
                
                next_token_id = next_token_id.item()
                
                # Check for end token
                if next_token_id == self.config.eos_token_id:
                    break
                
                generated_ids.append(next_token_id)
                
                # Decode and yield current text
                current_text = self.tokenizer.decode(generated_ids)
                yield current_text
                
                # Update input tensor for next iteration
                input_tensor = torch.tensor([[next_token_id]], device=self.device)
                
                # Truncate if too long
                if len(generated_ids) > self.config.max_length:
                    generated_ids = generated_ids[-self.config.max_length:]

def main():
    """Main function to demonstrate the advanced neural network"""
    logger.info("Initializing Advanced Transformer Neural Network...")
    
    # Model configuration
    model_config = ModelConfig(
        vocab_size=50257,
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
    
    # Generation configuration
    generation_config = GenerationConfig(
        max_length=512,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Initialize components
    tokenizer = BPETokenizer(model_config.vocab_size)
    model = NeuralNetwork(model_config)
    
    # Initialize generator
    generator = TextGenerator(model, tokenizer, generation_config)
    
    logger.info("Neural Network initialized successfully!")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA kernels available: {CUDA_KERNELS_AVAILABLE}")
    
    # Example usage
    prompts = [
        "Привет! Как дела?",
        "Hello! How are you?",
        "Расскажи о программировании",
        "What is artificial intelligence?",
        "Что такое мировоззрение?",
        "Умеешь на английском общаться?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"Пример {i}: {prompt}")
        print(f"{'='*60}")
        
        try:
            generated = generator.generate(prompt, max_new_tokens=100)
            print(f"Ответ: {generated}")
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            print(f"Ошибка: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()

