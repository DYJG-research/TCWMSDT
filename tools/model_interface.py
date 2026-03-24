#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model interface module

Provides a unified model invocation interface for API and local backends.
"""

import logging
import re
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True) -> str:
        """
        Generate text
        
        Args:
            prompt: input prompt
            max_tokens: maximum number of generated tokens
            temperature: sampling temperature
            clean_think: whether to remove <think> blocks (True unless Think completeness is needed)
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieve model metadata

        Returns:
            Dictionary of model information
        """
        pass
class APIModelInterface(ModelInterface):
    """API-backed model interface (OpenAI compatible)"""
    
    def __init__(self, api_url: str, model_name: str, api_key: str):
        """
        Initialize the API model interface
        
        Args:
            api_url: API endpoint
            model_name: model identifier
            api_key: API key
        """
        self.api_url = api_url
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize the OpenAI-compatible client
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )
        self._last_reasoning_text = ""
        self.diagnose_result={
            "reasoning_content":"",
            "treatment_plan":"",
            "precautions":""
        }

        logger.info(f"Initialized API model interface: {api_url} model: {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True) -> str:
        """
        Generate text via API (streaming to stdout)
        
        Args:
            prompt: input prompt
            max_tokens: maximum number of generated tokens
            temperature: sampling temperature
            clean_think: whether to strip <think> blocks
            
        Returns:
            Tuple of reasoning content and final content, honoring clean_think
        """
        logger.info(
            "="*25+"API Generate Parameters"+"="*25+"\n (model_name=%s, max_tokens=%s, temperature=%s, stream=True,clean_think=%s). Prompt:\n%s\n"+"="*50,
            self.model_name,
            max_tokens,
            temperature,
            clean_think,
            prompt,
        )
        full_text_parts = []
        reasoning_parts = []

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                try:
                    reasoning_delta = chunk.choices[0].delta.reasoning_content
                except Exception:
                    reasoning_delta = None
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                if delta:
                    print(delta, end="", flush=True)
                    full_text_parts.append(delta)
        except Exception as e:
            logger.exception("API call failed with error: {}".format(e))
            raise
        finally:
            print()
        
        reasoning_content = "".join(reasoning_parts).strip()
        content = "".join(full_text_parts)
        
        if reasoning_content:
            logger.info("="*25+"API model output(think_content)"+"="*25+"\n%s\n"+"="*50, reasoning_content or "[empty]")
        logger.info("="*25+"API model output(content)"+"="*25+"\n%s\n"+"="*50, content or "[empty]")

        if clean_think:################
            return "",self._clean_think_tags(content).strip() # return think, content
        return reasoning_content.strip(),content.strip()
    
    def _clean_think_tags(self, content: str) -> str:
        if "<think>" in content and "</think>" in content:
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return cleaned.strip()
        if "</think>" in content and "<think>" not in content:
            idx = content.find("</think>")
            return content[idx + len("</think>"):].strip()
        return content.strip()

    def get_last_reasoning(self) -> str:
        """
        Return the reasoning_content captured during the latest generate call.
        Empty string if the backend does not support reasoning traces.
        """
        return self._last_reasoning_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the API model"""
        return {
            "model_type": "api",
            "api_url": self.api_url,
            "model_name": self.model_name
        }

class LocalModelInterface(ModelInterface):
    """Local model interface"""
    
    def __init__(self, model_path: str, gpu_id: int = -1):
        """
        Initialize the local model interface (mirrors tuili.py loading logic)
        
        Args:
            model_path: path to the local model
            gpu_id: GPU identifier (controlled via CUDA_VISIBLE_DEVICES),
                   -1 for CPU, >=0 for GPU mode
        
        Note:
            When using a GPU, set CUDA_VISIBLE_DEVICES before running the script.
            Example: CUDA_VISIBLE_DEVICES=5 python tcwm_benchmark.py ...
        """
        self.model_path = model_path
        self.gpu_id = gpu_id
        
         # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True
            )
            
            if gpu_id >= 0 and torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.device = "cuda:0"  
                
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
                logger.info(f"Local model loaded: {model_path}")
                logger.info(f"  - CUDA_VISIBLE_DEVICES: {visible_devices}")
                logger.info(f"  - Device: {self.device}")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.device = "cpu"
                self.model = self.model.to(self.device)
                logger.info(f"Local model loaded: {model_path} device: cpu")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    def generate(self, prompt: str, max_tokens: int = 4096, 
                temperature: float = 0.0, clean_think: bool = True) -> str:
        """
        Generate text (with streaming output)
        
        Args:
            prompt: input prompt
            max_tokens: maximum number of generated tokens
            temperature: sampling temperature
            clean_think: whether to strip <think> blocks
            
        Returns:
            Generated text
        """
        try:
            # Encode input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move tensors to the appropriate device
            if self.device == "auto":
                inputs = inputs.to(self.model.device)
            else:
                inputs = inputs.to(self.device)
            
            # Create streamer for incremental output
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation kwargs
            generation_kwargs = {
                **inputs,
                'max_new_tokens': max_tokens,
                'temperature': temperature if temperature > 0 else None,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer,
            }
            
            # Generate on a background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Collect full text while printing to stdout
            generated_parts = []
            for new_text in streamer:
                print(new_text, end='', flush=True)
                generated_parts.append(new_text)
            
            # Wait for generation to finish
            thread.join()
            
            # Newline after streaming output
            print()
            
            # Full generated content
            content = ''.join(generated_parts).strip()
            logger.info("Local model raw output: %s", content)
            
            # Optionally strip <think> blocks
            if clean_think:
                content = self._clean_think_tags(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise
    
    def _clean_think_tags(self, content: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the local model"""
        return {
            "model_type": "local",
            "model_path": self.model_path,
            "device": self.device
        }
