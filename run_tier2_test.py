"""
Quick performance test runner for Tier 2 optimizations.
Tests Groq integration and VAD.
"""

import asyncio
import sys
from tests.test_tier2_performance import main

if __name__ == "__main__":
    print("Starting Tier 2 Performance Tests...")
    print("This will test: Groq Whisper, Groq Llama 3.1, VAD, and full streaming pipeline\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
