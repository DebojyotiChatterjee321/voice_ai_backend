"""
Performance tests for Tier 2 optimizations (Groq integration).
Tests the optimized pipeline with Groq services.
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from app.services.service_selector import service_selector
from app.services.vad import vad_service


class Tier2Metrics:
    """Track Tier 2 performance metrics."""
    
    def __init__(self):
        self.results = {
            "groq_stt": [],
            "groq_llm": [],
            "groq_llm_ttft": [],  # Time to first token
            "tts": [],
            "total": [],
            "vad_savings": []
        }
        self.tier1_baseline = {
            "stt": 350,  # ms
            "llm": 1300,  # ms
            "tts": 700,  # ms
            "total": 1800  # ms
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
        print("🚀 TIER 2 OPTIMIZATION PERFORMANCE RESULTS")
        print("="*60)
        
        # Individual components
        print("\n📊 Component Performance:")
        
        stt_avg = self.get_average('groq_stt')
        llm_avg = self.get_average('groq_llm')
        ttft_avg = self.get_average('groq_llm_ttft')
        tts_avg = self.get_average('tts')
        
        print(f"  STT (Groq Whisper):    {stt_avg*1000:>6.0f}ms (Target: <200ms) {'✅' if stt_avg < 0.2 else '⚠️'}")
        print(f"  LLM (Groq Llama):      {llm_avg*1000:>6.0f}ms (Target: <500ms) {'✅' if llm_avg < 0.5 else '⚠️'}")
        if ttft_avg > 0:
            print(f"    └─ Time to 1st token: {ttft_avg*1000:>6.0f}ms")
        print(f"  TTS (ElevenLabs):      {tts_avg*1000:>6.0f}ms (Target: <600ms) {'✅' if tts_avg < 0.6 else '⚠️'}")
        
        # VAD savings
        vad_avg = self.get_average('vad_savings')
        if vad_avg > 0:
            print(f"  VAD Silence Removal:   {vad_avg*1000:>6.0f}ms saved")
        
        # Total pipeline
        avg_total = self.get_average('total')
        print(f"\n🎯 Total Pipeline:       {avg_total*1000:>6.0f}ms")
        
        # Comparison with Tier 1
        tier1_total = self.tier1_baseline['total']
        current = avg_total * 1000
        tier2_improvement = ((tier1_total - current) / tier1_total) * 100
        
        print(f"\n📈 Tier 2 Improvement:")
        print(f"  Tier 1 Baseline:       {tier1_total:>6.0f}ms")
        print(f"  Tier 2 Current:        {current:>6.0f}ms")
        print(f"  Improvement:           {tier2_improvement:>6.1f}%")
        print(f"  Time saved:            {tier1_total - current:>6.0f}ms")
        
        # Total improvement from original baseline
        original_baseline = 5042
        total_improvement = ((original_baseline - current) / original_baseline) * 100
        
        print(f"\n🎊 Total Improvement from Baseline:")
        print(f"  Original Baseline:     {original_baseline:>6.0f}ms")
        print(f"  Current (Tier 2):      {current:>6.0f}ms")
        print(f"  Total Improvement:     {total_improvement:>6.1f}%")
        print(f"  Total Time Saved:      {original_baseline - current:>6.0f}ms")
        
        # Target assessment
        target = 1200
        if current <= target:
            print(f"\n✅ SUCCESS: Met Tier 2 target of <{target}ms!")
        else:
            print(f"\n⚠️  PROGRESS: {current - target:.0f}ms over target")
        
        # Component breakdown comparison
        print(f"\n📊 Component Comparison (Tier 1 → Tier 2):")
        print(f"  STT:  {self.tier1_baseline['stt']}ms → {stt_avg*1000:.0f}ms ({((self.tier1_baseline['stt'] - stt_avg*1000)/self.tier1_baseline['stt']*100):.0f}% faster)")
        print(f"  LLM:  {self.tier1_baseline['llm']}ms → {llm_avg*1000:.0f}ms ({((self.tier1_baseline['llm'] - llm_avg*1000)/self.tier1_baseline['llm']*100):.0f}% faster)")
        print(f"  TTS:  {self.tier1_baseline['tts']}ms → {tts_avg*1000:.0f}ms ({((self.tier1_baseline['tts'] - tts_avg*1000)/self.tier1_baseline['tts']*100):.0f}% faster)")
        
        print("="*60 + "\n")


async def test_groq_stt(metrics: Tier2Metrics):
    """Test Groq STT performance."""
    print("\n🎤 Testing Groq STT (Whisper-large-v3)...")
    
    # Generate test audio (2 seconds)
    sample_rate = 16000
    duration = 2.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    # Run multiple tests
    for i in range(3):
        start = time.time()
        result = await service_selector.transcribe_audio(test_audio)
        latency = time.time() - start
        
        metrics.add_result("groq_stt", latency)
        service = result.get("service", "unknown")
        print(f"  Test {i+1}: {latency*1000:.0f}ms [{service}] - {'✅' if latency < 0.2 else '⚠️'}")
    
    avg = metrics.get_average("groq_stt")
    print(f"  Average: {avg*1000:.0f}ms (Target: <200ms) - {'✅ PASS' if avg < 0.2 else '❌ NEEDS WORK'}")


async def test_groq_llm(metrics: Tier2Metrics):
    """Test Groq LLM performance."""
    print("\n🤖 Testing Groq LLM (Llama 3.1 70B)...")
    
    test_queries = [
        "What is your return policy?",
        "How do I track my order?",
        "What payment methods do you accept?",
    ]
    
    for i, query in enumerate(test_queries):
        start = time.time()
        result = await service_selector.generate_response(query)
        latency = time.time() - start
        
        metrics.add_result("groq_llm", latency)
        
        ttft = result.get("time_to_first_token", 0)
        if ttft > 0:
            metrics.add_result("groq_llm_ttft", ttft)
        
        service = result.get("service", "unknown")
        print(f"  Query {i+1}: {latency*1000:.0f}ms [{service}] (TTFT: {ttft*1000:.0f}ms) - {'✅' if latency < 0.5 else '⚠️'}")
    
    avg = metrics.get_average("groq_llm")
    ttft_avg = metrics.get_average("groq_llm_ttft")
    print(f"  Average: {avg*1000:.0f}ms (Target: <500ms) - {'✅ PASS' if avg < 0.5 else '❌ NEEDS WORK'}")
    if ttft_avg > 0:
        print(f"  Avg TTFT: {ttft_avg*1000:.0f}ms (Lower is better)")


async def test_vad(metrics: Tier2Metrics):
    """Test VAD performance."""
    print("\n🔊 Testing VAD (Voice Activity Detection)...")
    
    # Generate audio with silence padding
    sample_rate = 16000
    silence_duration = 1.0  # 1 second silence
    speech_duration = 2.0   # 2 seconds speech
    
    silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
    speech = np.random.randn(int(sample_rate * speech_duration)).astype(np.float32) * 0.1
    
    # Audio with silence padding
    padded_audio = np.concatenate([silence, speech, silence])
    
    start = time.time()
    trimmed = await vad_service.trim_silence(padded_audio)
    vad_time = time.time() - start
    
    original_duration = len(padded_audio) / sample_rate
    trimmed_duration = len(trimmed) / sample_rate
    saved = original_duration - trimmed_duration
    
    metrics.add_result("vad_savings", saved)
    
    print(f"  Original duration: {original_duration:.2f}s")
    print(f"  Trimmed duration:  {trimmed_duration:.2f}s")
    print(f"  Silence removed:   {saved:.2f}s ({saved/original_duration*100:.1f}%)")
    print(f"  VAD processing:    {vad_time*1000:.0f}ms")
    
    if saved > 0.5:
        print(f"  ✅ VAD effectively removed silence")
    else:
        print(f"  ⚠️ VAD may not be enabled or audio has no silence")


async def test_full_pipeline_tier2(metrics: Tier2Metrics):
    """Test full Tier 2 pipeline."""
    print("\n🎯 Testing Full Tier 2 Pipeline (Groq STT → Groq LLM → TTS)...")
    
    # Generate test audio
    sample_rate = 16000
    duration = 2.0
    test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    # Run pipeline tests
    for i in range(3):
        start = time.time()
        
        # STT
        stt_start = time.time()
        stt_result = await service_selector.transcribe_audio(test_audio)
        stt_time = time.time() - stt_start
        
        # LLM
        llm_start = time.time()
        llm_result = await service_selector.generate_response("What is my order status?")
        llm_time = time.time() - llm_start
        
        # TTS
        tts_start = time.time()
        tts_result = await service_selector.synthesize_speech(llm_result.get("response", "Hello"))
        tts_time = time.time() - tts_start
        
        total = time.time() - start
        metrics.add_result("total", total)
        metrics.add_result("tts", tts_time)
        
        stt_svc = stt_result.get("service", "?")
        llm_svc = llm_result.get("service", "?")
        tts_svc = tts_result.get("service", "?")
        ttft = llm_result.get("time_to_first_token", 0)
        
        print(f"  Run {i+1}: {total*1000:.0f}ms")
        print(f"    STT: {stt_time*1000:.0f}ms [{stt_svc}]")
        print(f"    LLM: {llm_time*1000:.0f}ms [{llm_svc}] (TTFT: {ttft*1000:.0f}ms)")
        print(f"    TTS: {tts_time*1000:.0f}ms [{tts_svc}]")
    
    avg = metrics.get_average("total")
    print(f"  Average: {avg*1000:.0f}ms (Target: <1200ms) - {'✅ PASS' if avg < 1.2 else '❌ NEEDS WORK'}")


async def main():
    """Run all Tier 2 performance tests."""
    print("\n" + "="*60)
    print("🚀 TIER 2.5 OPTIMIZATION PERFORMANCE TEST")
    print("   (Groq Integration + Cartesia TTS + VAD)")
    print("="*60)
    print("\nInitializing services...")
    
    # Initialize services
    try:
        await service_selector.initialize()
        await vad_service.initialize()
        
        # Check which services are active
        status = await service_selector.get_service_status()
        print(f"\n✅ Services initialized:")
        print(f"   STT: {status['stt_service']}")
        print(f"   LLM: {status['llm_service']}")
        print(f"   TTS: {status['tts_service']}")
        print(f"   Groq STT: {'✅' if status['groq_stt_available'] else '❌'}")
        print(f"   Groq LLM: {'✅' if status['groq_llm_available'] else '❌'}")
        print(f"   Cartesia TTS: {'✅' if status['cartesia_tts_available'] else '❌'}")
        print(f"   VAD: {'✅' if status['vad_enabled'] else '❌'}")
        print()
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create metrics tracker
    metrics = Tier2Metrics()
    
    # Run tests
    try:
        await test_groq_stt(metrics)
        await test_groq_llm(metrics)
        await test_vad(metrics)
        await test_full_pipeline_tier2(metrics)
        
        # Print summary
        metrics.print_summary()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
