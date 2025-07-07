"""
Performance Monitoring Demonstration for Sovereign AI Agent

This demo shows how to integrate the performance monitoring framework
with the Sovereign AI system for real-time performance tracking and optimization.

Features demonstrated:
- Real-time response time tracking for AI model inference
- Voice interface reliability monitoring
- System resource monitoring (CPU, memory, disk, network)
- Automated alert generation for performance issues
- Performance optimization suggestions
- Metrics export and analysis
"""

import asyncio
import time
import logging
from pathlib import Path
import json

# Import Sovereign AI components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.sovereign.performance_monitor import (
    PerformanceMonitor, QueryType, ResponseTimeTracker, 
    track_response_time, PerformanceThresholds
)
from src.sovereign.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)


class SovereignAISimulator:
    """
    Simulates Sovereign AI operations for performance monitoring demo
    """
    
    def __init__(self):
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            enable_gpu_monitoring=True,
            enable_real_time_alerts=True,
            metrics_retention_days=30
        )
        
        # Set custom thresholds
        self.performance_monitor.thresholds = PerformanceThresholds(
            talker_max_response_ms=2000,
            thinker_max_response_ms=30000,
            max_memory_usage_percent=80.0,
            min_voice_reliability_percent=95.0
        )
        
        # Set up alert callbacks
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        # Start continuous monitoring
        self.performance_monitor.start_monitoring(interval_seconds=5.0)
        
        logger.info("Sovereign AI Simulator initialized with performance monitoring")
    
    def _handle_performance_alert(self, alert):
        """Handle performance alerts"""
        logger.warning(f"Performance Alert: {alert.message}")
        
        # In a real system, you might:
        # - Send notifications to administrators
        # - Trigger automatic optimization routines
        # - Log to external monitoring systems
        # - Adjust system parameters dynamically
    
    def simulate_talker_response(self, query: str, complexity: str = "simple") -> dict:
        """
        Simulate Talker model response with performance tracking
        """
        start_time = time.time()
        logger.info(f"Processing Talker query: {query[:50]}...")
        
        # Simulate processing time based on complexity
        if complexity == "simple":
            time.sleep(0.8)  # 800ms - good performance
            query_type = QueryType.TALKER_SIMPLE
            model_used = "gemma2:9b"
        elif complexity == "complex":
            time.sleep(1.5)  # 1.5s - acceptable performance
            query_type = QueryType.TALKER_SIMPLE
            model_used = "gemma2:9b"
        else:  # slow
            time.sleep(3.0)  # 3s - slow performance (will trigger alert)
            query_type = QueryType.TALKER_SIMPLE
            model_used = "gemma2:9b"
        
        end_time = time.time()
        
        # Simulate token generation
        token_count = len(query.split()) * 3  # Rough estimate
        
        # Track performance
        self.performance_monitor.track_response_time(
            start_time=start_time,
            end_time=end_time,
            query_type=query_type,
            model_used=model_used,
            token_count=token_count,
            success=True
        )
        
        return {
            "response": f"Generated response to: {query}",
            "token_count": token_count,
            "model": model_used,
            "query_type": query_type
        }
    
    def simulate_thinker_response(self, query: str) -> dict:
        """Simulate Thinker model response (reasoning-heavy)"""
        with ResponseTimeTracker(
            self.performance_monitor, 
            QueryType.THINKER_REASONING, 
            "deepseek-r1:14b"
        ) as tracker:
            logger.info(f"Processing Thinker reasoning query: {query[:50]}...")
            
            # Simulate complex reasoning
            time.sleep(5.0)  # 5 seconds of reasoning
            
            token_count = len(query.split()) * 10  # More tokens for reasoning
            tracker.set_token_count(token_count)
            
            return {
                "response": f"Detailed reasoning response to: {query}",
                "reasoning_steps": ["Step 1", "Step 2", "Step 3"],
                "token_count": token_count
            }
    
    def simulate_voice_operations(self):
        """Simulate voice recognition and synthesis operations"""
        logger.info("Simulating voice operations...")
        
        # Simulate voice recognition
        self.performance_monitor.track_voice_reliability(
            operation_type="recognition",
            success=True,
            duration_ms=1200.0,
            audio_quality_score=0.92,
            confidence_score=0.89
        )
        
        # Simulate voice synthesis
        self.performance_monitor.track_voice_reliability(
            operation_type="synthesis",
            success=True,
            duration_ms=800.0,
            audio_quality_score=0.95,
            confidence_score=0.94
        )
        
        # Simulate occasional failure
        self.performance_monitor.track_voice_reliability(
            operation_type="recognition",
            success=False,
            duration_ms=2500.0,
            audio_quality_score=0.3,  # Poor quality
            confidence_score=0.2
        )
    
    async def simulate_rag_search(self, query: str) -> dict:
        """Simulate RAG system search operations"""
        start_time = time.time()
        
        logger.info(f"Processing RAG search: {query[:50]}...")
        
        # Simulate vector search and retrieval
        await asyncio.sleep(0.5)  # 500ms for search
        
        end_time = time.time()
        
        # Track the RAG search performance
        metric = self.performance_monitor.track_response_time(
            start_time=start_time,
            end_time=end_time,
            query_type=QueryType.RAG_SEARCH,
            model_used="bge-large-en-v1.5",
            token_count=50,
            success=True
        )
        
        return {
            "search_results": [
                {"score": 0.92, "content": "Relevant result 1"},
                {"score": 0.85, "content": "Relevant result 2"}
            ],
            "search_time_ms": metric.duration_ms
        }
    
    def get_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        return self.performance_monitor.get_performance_summary(hours=1)
    
    def export_metrics(self) -> str:
        """Export performance metrics to file"""
        return self.performance_monitor.export_metrics(hours=24)
    
    def cleanup(self):
        """Clean up monitoring resources"""
        self.performance_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")


async def run_demo():
    """Run the performance monitoring demonstration"""
    logger.info("Starting Sovereign AI Performance Monitoring Demo")
    
    # Initialize the simulator
    simulator = SovereignAISimulator()
    
    try:
        # Demo 1: Test various response scenarios
        logger.info("\nDemo 1: Response Time Monitoring")
        
        # Fast response
        result1 = simulator.simulate_talker_response(
            "What is the weather like today?", 
            complexity="simple"
        )
        logger.info(f"Fast response generated: {len(result1['response'])} chars")
        
        # Medium response
        result2 = simulator.simulate_talker_response(
            "Explain the concept of machine learning in detail", 
            complexity="complex"
        )
        logger.info(f"Medium response generated: {len(result2['response'])} chars")
        
        # Slow response (will trigger alert)
        result3 = simulator.simulate_talker_response(
            "Write a comprehensive analysis of quantum computing", 
            complexity="slow"
        )
        logger.info(f"Slow response generated: {len(result3['response'])} chars")
        
        # Demo 2: Reasoning tasks
        logger.info("\nDemo 2: Thinker Model Monitoring")
        
        reasoning_result = simulator.simulate_thinker_response(
            "Analyze the economic implications of AI automation on employment"
        )
        logger.info(f"Reasoning response: {len(reasoning_result['reasoning_steps'])} steps")
        
        # Demo 3: Voice interface monitoring
        logger.info("\nDemo 3: Voice Interface Monitoring")
        simulator.simulate_voice_operations()
        
        # Demo 4: RAG system monitoring
        logger.info("\nDemo 4: RAG Search Monitoring")
        
        rag_result = await simulator.simulate_rag_search(
            "Find information about neural network architectures"
        )
        logger.info(f"RAG search completed: {len(rag_result['search_results'])} results")
        
        # Let monitoring collect some data
        await asyncio.sleep(3)
        
        # Demo 5: Performance analysis
        logger.info("\nDemo 5: Performance Analysis")
        
        # Get current system status
        status = simulator.performance_monitor.get_current_status()
        logger.info(f"Current memory usage: {status['system']['memory_percent']:.1f}%")
        logger.info(f"Current CPU usage: {status['system']['cpu_percent']:.1f}%")
        
        # Generate performance summary
        report = simulator.get_performance_report()
        
        logger.info(f"\nPerformance Summary (Last Hour):")
        logger.info(f"  System: {report['system_info']['platform']}")
        logger.info(f"  GPU Available: {report['system_info']['gpu_available']}")
        
        if 'response_times' in report:
            for query_type, stats in report['response_times'].items():
                logger.info(f"  {query_type.replace('_', ' ').title()}:")
                logger.info(f"    Requests: {stats['total_requests']}")
                logger.info(f"    Success Rate: {stats['success_rate']:.1f}%")
                logger.info(f"    Avg Response: {stats['avg_duration_ms']:.0f}ms")
        
        if 'voice_reliability' in report:
            voice_stats = report['voice_reliability']
            logger.info(f"  Voice Reliability: {voice_stats['overall_success_rate']:.1f}%")
        
        # Show alerts
        if report['alerts']:
            logger.info(f"\nActive Alerts: {len(report['alerts'])}")
            for alert in report['alerts'][-3:]:  # Show last 3 alerts
                logger.warning(f"  {alert['alert_level'].upper()}: {alert['message']}")
        
        # Show optimization suggestions
        if report['optimization_suggestions']:
            logger.info(f"\nOptimization Suggestions:")
            for suggestion in report['optimization_suggestions'][:3]:  # Show top 3
                logger.info(f"  - {suggestion}")
        
        # Demo 6: Metrics export
        logger.info("\nDemo 6: Metrics Export")
        
        export_path = simulator.export_metrics()
        logger.info(f"Metrics exported to: {export_path}")
        
        # Show a sample of exported data
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        logger.info(f"Export contains {len(export_data['response_times'])} response time metrics")
        logger.info(f"Time period: {export_data['time_period_hours']} hours")
        
        logger.info("\nPerformance Monitoring Demo Complete!")
        logger.info("Key features demonstrated:")
        logger.info("  - Real-time response time tracking")
        logger.info("  - Voice interface reliability monitoring")
        logger.info("  - System resource monitoring")
        logger.info("  - Automated alert generation")
        logger.info("  - Performance optimization suggestions")
        logger.info("  - Comprehensive metrics export")
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
    
    finally:
        # Clean up
        simulator.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo()) 