"""
Entry point script for the ask_llm application.
This allows running the app directly from the project root.
"""
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from ask_llm.main import main

if __name__ == "__main__":
    main()
