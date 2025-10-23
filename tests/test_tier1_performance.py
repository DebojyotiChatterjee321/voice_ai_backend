"""
Performance tests for Tier 1 optimizations.
Tests the optimized pipeline against performance targets.
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.services import stt_service, llm_service, tts_service, initialize_all_services
from app.pipecat import voice_pipeline


class PerformanceMetrics:
    """Track performance metrics."""
    
    def __init__(self):
        self.results = {
            "stt": [],
            "llm": [],
            "tts": [],
            "total": [],
            "cached_llm": []
        }
    
    def add_result(self, category: str, latency: float):
        """Add a performance result."""
        self.results[category].append(latency)
    
    def get_average(self, category: str) -> float:
        """Get average latency for a category."""
        if not self.results[category]:
            return 0.0
        return sum(self.results[category]) / len(self.results[category])
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("🎯 TIER 1 OPTIMIZATION PERFORMANCE RESULTS")
        print("="*60)
        
        # Individual components
        print("\n📊 Component Performance:")
        print(f"  STT (faster-whisper):  {self.get_average('stt')*1000:>6.0f}ms (Target: <400ms)")
        print(f"  LLM (optimized):       {self.get_average('llm')*1000:>6.0f}ms (Target: <1500ms)")
        print(f"  LLM (cached):          {self.get_average('cached_llm')*1000:>6.0f}ms (Target: <50ms)")
        print(f"  TTS (ElevenLabs):      {self.get_average('tts')*1000:>6.0f}ms (Target: <800ms)")
        
        # Total pipeline
        avg_total = self.get_average('total')
        print(f"\n🚀 Total Pipeline:       {avg_total*1000:>6.0f}ms")
        
        # Comparison
        baseline = 5042  # Original baseline
        current = avg_total * 1000
        improvement = ((baseline - current) / baseline) * 100
        
        print(f"\n📈 Improvement:")
        print(f"  Baseline:              {baseline:>6.0f}ms")
        print(f"  Current:               {current:>6.0f}ms")
        print(f"  Improvement:           {improvement:>6.1f}%")
        print(f"  Time saved:            {baseline - current:>6.0f}ms")
        
        # Target assessment
        target = 2000
        if current <= target:
            print(f"\n✅ SUCCESS: Met Tier 1 target of <{target}ms!")
        else:
            print(f"\n⚠️  PROGRESS: {target - current:.0f}ms away from target")
        
        print("="*60 + "\n")


async def test_stt_performance(metrics: PerformanceMetrics):
    """Test STT performance with faster-whisper."""
    print("\n🎤 Testing STT (faster-whisper)...")
    
    # Generate test audio (1 second of simulated speech)
    sample_rate = 16000
    duration = 1.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    # Run multiple tests
    for i in range(3):
        start = time.time()
        result = await stt_service.transcribe_audio(test_audio)
        latency = time.time() - start
        
        metrics.add_result("stt", latency)
        print(f"  Test {i+1}: {latency*1000:.0f}ms - {'✅' if latency < 0.4 else '⚠️'}")
    
    avg = metrics.get_average("stt")
    print(f"  Average: {avg*1000:.0f}ms (Target: <400ms) - {'✅ PASS' if avg < 0.4 else '❌ NEEDS WORK'}")


async def test_llm_performance(metrics: PerformanceMetrics):
    """Test LLM performance with optimizations."""
    print("\n🤖 Testing LLM (optimized + caching)...")
    
    test_queries = [
        "What is your return policy?",
        "How do I track my order?",
        "What payment methods do you accept?",
    ]
    
    # First run (no cache)
    print("  First run (no cache):")
    for i, query in enumerate(test_queries):
        start = time.time()
        result = await llm_service.generate_response(query)
        latency = time.time() - start
        
        metrics.add_result("llm", latency)
        cached = result.get("cached", False)
        print(f"    Query {i+1}: {latency*1000:.0f}ms - {'[CACHED]' if cached else '[NEW]'}")
    
    # Second run (should hit cache)
    print("  Second run (cache test):")
    for i, query in enumerate(test_queries):
        start = time.time()
        result = await llm_service.generate_response(query)
        latency = time.time() - start
        
        if result.get("cached", False):
            metrics.add_result("cached_llm", latency)
            print(f"    Query {i+1}: {latency*1000:.0f}ms - ✅ [CACHED]")
        else:
            print(f"    Query {i+1}: {latency*1000:.0f}ms - ⚠️ [CACHE MISS]")
    
    avg = metrics.get_average("llm")
    avg_cached = metrics.get_average("cached_llm")
    print(f"  Average (uncached): {avg*1000:.0f}ms (Target: <1500ms) - {'✅ PASS' if avg < 1.5 else '❌ NEEDS WORK'}")
    if avg_cached > 0:
        print(f"  Average (cached):   {avg_cached*1000:.0f}ms (Target: <50ms) - {'✅ PASS' if avg_cached < 0.05 else '⚠️'}")


async def test_tts_performance(metrics: PerformanceMetrics):
    """Test TTS performance."""
    print("\n🔊 Testing TTS (ElevenLabs)...")
    
    test_texts = [
        "Hello, how can I help you today?",
        "Your order has been shipped.",
        "Thank you for contacting support.",
    ]
    
    for i, text in enumerate(test_texts):
        start = time.time()
        result = await tts_service.synthesize_speech(text)
        latency = time.time() - start
        
        metrics.add_result("tts", latency)
        success = result.get("success", False)
        print(f"  Test {i+1}: {latency*1000:.0f}ms - {'✅' if success and latency < 0.8 else '⚠️'}")
    
    avg = metrics.get_average("tts")
    print(f"  Average: {avg*1000:.0f}ms (Target: <800ms) - {'✅ PASS' if avg < 0.8 else '❌ NEEDS WORK'}")


async def test_full_pipeline(metrics: PerformanceMetrics):
    """Test full end-to-end pipeline."""
    print("\n🎯 Testing Full Pipeline (STT → LLM → TTS)...")
    
    # Generate test audio
    sample_rate = 16000
    duration = 2.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    # Run pipeline tests
    for i in range(3):
        start = time.time()
        
        # STT
        stt_start = time.time()
        stt_result = await stt_service.transcribe_audio(test_audio)
        stt_time = time.time() - stt_start
        
        # LLM
        llm_start = time.time()
        llm_result = await llm_service.generate_response("What is my order status?")
        llm_time = time.time() - llm_start
        
        # TTS
        tts_start = time.time()
        tts_result = await tts_service.synthesize_speech(llm_result.get("response", "Hello"))
        tts_time = time.time() - tts_start
        
        total = time.time() - start
        metrics.add_result("total", total)
        
        cached = llm_result.get("cached", False)
        print(f"  Run {i+1}: {total*1000:.0f}ms (STT: {stt_time*1000:.0f}ms, LLM: {llm_time*1000:.0f}ms{'[C]' if cached else ''}, TTS: {tts_time*1000:.0f}ms)")
    
    avg = metrics.get_average("total")
    print(f"  Average: {avg*1000:.0f}ms (Target: <2000ms) - {'✅ PASS' if avg < 2.0 else '❌ NEEDS WORK'}")


async def main():
    """Run all performance tests."""
    print("\n" + "="*60)
    print("🚀 TIER 1 OPTIMIZATION PERFORMANCE TEST")
    print("="*60)
    print("\nInitializing services...")
    
    # Initialize services
    try:
        await initialize_all_services()
        print("✅ All services initialized\n")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return
    
    # Create metrics tracker
    metrics = PerformanceMetrics()
    
    # Run tests
    try:
        await test_stt_performance(metrics)
        await test_llm_performance(metrics)
        await test_tts_performance(metrics)
        await test_full_pipeline(metrics)
        
        # Print summary
        metrics.print_summary()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
