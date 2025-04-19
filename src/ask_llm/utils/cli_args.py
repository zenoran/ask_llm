"""
Command-line argument utilities for SesameAI applications.

This module provides common utilities for handling command-line arguments
across multiple applications to avoid code duplication.
"""
import os
import argparse
from pathlib import Path

# Import the global config instance directly
from ask_llm.utils.config import config
from ask_llm.utils.ollama_utils import find_matching_model
from ask_llm.model_manager import refresh_ollama_models

def setup_common_args_and_config(args: argparse.Namespace):
    """
    Applies common command-line arguments (like --model, --voice) to the global config,
    potentially overriding environment variables or models.yaml defaults.
    This function should be called *after* parsing args but *before* initializing
    components that rely on these default values (like AskLLM or TTSBaseApp).
    """
    # Handle model refresh if requested
    if hasattr(args, 'refresh_models') and args.refresh_models:
        if args.refresh_models == 'ollama':
            refresh_ollama_models(config)
        else:
            print(f"[yellow]Unknown model type to refresh: {args.refresh_models}[/yellow]")
        # Exit after refresh
        return

    # Update default model *alias* if --model is provided
    # The actual resolution and validation happen later.
    if args.model:
        print(f"CLI argument --model '{args.model}' provided. Setting as default requested alias.")
        config.DEFAULT_MODEL_ALIAS = args.model
    elif config.VERBOSE:
        # Print the default being used if --model wasn't specified
        print(f"No --model specified, using default alias from config/env: {config.DEFAULT_MODEL_ALIAS}")

    # Update default voice if --voice is provided
    if hasattr(args, 'voice') and args.voice:
        print(f"CLI argument --voice '{args.voice}' provided. Setting as default voice.")
        config.DEFAULT_VOICE = args.voice
    elif config.VERBOSE:
        print(f"No --voice specified, using default voice from config/env: {config.DEFAULT_VOICE}")

    # Note: Setting config.VERBOSE is handled separately in the main scripts
    # after calling this function, to ensure CLI flag overrides .env.

def add_common_args(parser: argparse.ArgumentParser):
    """Add common command-line arguments to a parser."""
    parser.add_argument("-m", "--model", help="Choose the model to use (supports partial matching)")
    parser.add_argument("-v", "--voice", help="Choose the voice to use for TTS")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--refresh-models", choices=['ollama'], help="Refresh available models from the specified source") 