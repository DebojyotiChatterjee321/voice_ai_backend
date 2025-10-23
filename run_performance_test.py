"""
Quick performance test runner for Tier 1 optimizations.
Run this to validate the optimizations are working.
"""

import asyncio
import sys
from tests.test_tier1_performance import main

if __name__ == "__main__":
    print("Starting Tier 1 Performance Tests...")
    print("This will test: faster-whisper, LLM streaming, caching, and parallel DB queries\n")
    
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
