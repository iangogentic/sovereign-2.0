#!/usr/bin/env python3
"""
Performance Dashboard Demo

This demo showcases the integration of real-time performance monitoring
with the Sovereign AI GUI, including:
- Live system metrics display
- Performance charts
- Alert notifications
- Performance tracking integration

Run this demo to see the performance monitoring in action.
"""

import asyncio
import time
import threading
import random
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try direct imports, then fallback to package imports
try:
    from sovereign.config import Config
    from sovereign.logger import setup_logger
    from sovereign.performance_monitor import (
        PerformanceMonitor, AlertLevel, MetricType, QueryType
    )
except ImportError:
    # Fallback to src imports
    from src.sovereign.config import Config
    from src.sovereign.logger import setup_logger
    from src.sovereign.performance_monitor import (
        PerformanceMonitor, AlertLevel, MetricType, QueryType
    )


class PerformanceDashboardDemo:
    """Demonstrates the performance dashboard integration"""
    
    def __init__(self):
        """Initialize the demo"""
        self.config = Config()
        self.logger = setup_logger("performance_demo", "INFO")
        self.performance_monitor = None
        self.demo_running = False
        
    async def setup_performance_monitoring(self):
        """Setup performance monitoring with demo configuration"""
        print("🎯 Setting up performance monitoring...")
        
        # Initialize performance monitor with correct parameters
        self.performance_monitor = PerformanceMonitor(
            config_path=None,
            enable_gpu_monitoring=True,
            enable_real_time_alerts=True,
            metrics_retention_days=30
        )
        
        # Add alert callback
        self.performance_monitor.add_alert_callback(self._on_performance_alert)
        
        # Start monitoring with frequent updates for demo
        self.performance_monitor.start_monitoring(interval_seconds=1.0)
        
        print("✅ Performance monitoring started")
        
    def _on_performance_alert(self, alert):
        """Handle performance alerts"""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️", 
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🔥"
        }
        
        emoji = level_emoji.get(alert.alert_level, "📢")
        timestamp = alert.timestamp.strftime("%H:%M:%S")
        
        print(f"\n{emoji} ALERT [{timestamp}]: {alert.message}")
        if alert.suggested_action:
            print(f"   💡 Suggestion: {alert.suggested_action}")
        print()
        
    def simulate_ai_queries(self):
        """Simulate AI query processing to generate performance data"""
        query_types = [
            (QueryType.TALKER_SIMPLE, "talker", 800, 1200),
            (QueryType.TALKER_COMPLEX, "talker", 1500, 2500),
            (QueryType.THINKER_REASONING, "thinker", 3000, 6000),
            (QueryType.RAG_SEARCH, "rag", 400, 800),
            (QueryType.VOICE_PROCESSING, "voice", 200, 500)
        ]
        
        query_count = 0
        
        while self.demo_running:
            # Select random query type
            query_type, model, min_time, max_time = random.choice(query_types)
            
            # Simulate processing time
            processing_time = random.uniform(min_time, max_time)
            
            # Occasionally simulate slow responses or failures
            success = True
            if random.random() < 0.05:  # 5% chance of failure
                success = False
                processing_time *= 2  # Failed queries take longer
            elif random.random() < 0.1:  # 10% chance of slow response
                processing_time *= 3  # Very slow response
            
            # Track the response
            self.performance_monitor.track_response_time(
                start_time=time.time() - processing_time/1000,
                end_time=time.time(),
                query_type=query_type,
                model_used=model,
                success=success
            )
            
            query_count += 1
            print(f"📝 Processed query #{query_count}: {query_type.value} "
                  f"({processing_time:.0f}ms, {'✅' if success else '❌'})")
            
            # Wait before next query (vary the interval)
            time.sleep(random.uniform(0.5, 2.0))
            
    def display_performance_data(self):
        """Display current performance data in console format"""
        while self.demo_running:
            try:
                # Get current status and summary
                status = self.performance_monitor.get_current_status()
                summary = self.performance_monitor.get_performance_summary()
                
                # Clear screen (Windows/Unix compatible)
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("=" * 80)
                print("🎛️  SOVEREIGN AI - PERFORMANCE DASHBOARD DEMO")
                print("=" * 80)
                print()
                
                # System metrics - Fix API structure
                print("🖥️  SYSTEM METRICS")
                print("-" * 40)
                if 'system' in status:
                    sys_info = status['system']
                    print(f"CPU Usage:      {sys_info.get('cpu_percent', 0):.1f}%")
                    print(f"Memory Usage:   {sys_info.get('memory_percent', 0):.1f}%")
                    print(f"Disk Usage:     {sys_info.get('disk_usage_percent', 0):.1f}%")
                else:
                    print("CPU Usage:      0.0%")
                    print("Memory Usage:   0.0%")
                    print("Disk Usage:     0.0%")
                
                # GPU metrics - Fix API structure
                if 'gpu' in status and 'error' not in status['gpu']:
                    gpu_info = status['gpu']
                    print(f"GPU Memory:     {gpu_info.get('memory_percent', 0):.1f}%")
                    print(f"GPU Device:     {gpu_info.get('device_name', 'Unknown')}")
                
                # Monitoring status
                monitoring_active = status.get('monitoring_active', False)
                print(f"Monitoring:     {'✅ Active' if monitoring_active else '❌ Inactive'}")
                print()
                
                # Performance metrics - Fix API structure  
                print("⚡ AI PERFORMANCE METRICS")
                print("-" * 40)
                
                total_requests = 0
                avg_response_time = 0
                success_rate = 0
                
                # Calculate totals from response_times structure
                if 'response_times' in summary and summary['response_times']:
                    response_data = summary['response_times']
                    all_requests = []
                    all_durations = []
                    all_successes = []
                    
                    for query_type, data in response_data.items():
                        all_requests.append(data.get('total_requests', 0))
                        all_durations.append(data.get('avg_duration_ms', 0))
                        all_successes.append(data.get('success_rate', 0))
                    
                    if all_requests:
                        total_requests = sum(all_requests)
                        avg_response_time = sum(all_durations) / len(all_durations) if all_durations else 0
                        success_rate = sum(all_successes) / len(all_successes) if all_successes else 0
                
                print(f"Avg Response:   {avg_response_time:.0f}ms")
                print(f"Total Requests: {total_requests}")
                print(f"Success Rate:   {success_rate:.1f}%")
                
                # Voice reliability - Fix API structure
                if 'voice_reliability' in summary and summary['voice_reliability']:
                    voice_data = summary['voice_reliability']
                    for operation, data in voice_data.items():
                        reliability = data.get('reliability_percent', 0)
                        print(f"Voice {operation.title()}: {reliability:.1f}%")
                
                # Recent metrics count
                metrics_count = status.get('metrics_collected', 0)
                alerts_count = status.get('recent_alerts_count', 0)
                print(f"Metrics Stored: {metrics_count}")
                print(f"Recent Alerts:  {alerts_count}")
                print()
                
                # Performance status indicators
                print("🚦 STATUS INDICATORS")
                print("-" * 40)
                
                # Response time status
                if avg_response_time > 5000:
                    rt_status = "🔴 SLOW"
                elif avg_response_time > 2000:
                    rt_status = "🟡 MODERATE"
                else:
                    rt_status = "🟢 FAST"
                print(f"Response Time:  {rt_status}")
                
                # Success rate status
                if success_rate < 95:
                    sr_status = "🔴 POOR"
                elif success_rate < 99:
                    sr_status = "🟡 GOOD"
                else:
                    sr_status = "🟢 EXCELLENT"
                print(f"Success Rate:   {sr_status}")
                
                # System load status
                cpu = status.get('system', {}).get('cpu_percent', 0)
                memory = status.get('system', {}).get('memory_percent', 0)
                max_load = max(cpu, memory)
                if max_load > 90:
                    load_status = "🔴 HIGH"
                elif max_load > 70:
                    load_status = "🟡 MEDIUM"
                else:
                    load_status = "🟢 LOW"
                print(f"System Load:    {load_status}")
                print()
                
                print("💡 This dashboard shows real-time performance data")
                print(f"   Demo will auto-close, monitoring: {monitoring_active}")
                print("=" * 80)
                
            except Exception as e:
                print(f"Error displaying performance data: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(2.0)  # Update every 2 seconds
    
    async def run_demo(self):
        """Run the complete performance dashboard demo"""
        print("🚀 Starting Performance Dashboard Demo...")
        print("   This demo simulates AI query processing and displays")
        print("   real-time performance metrics as they would appear")
        print("   in the Sovereign AI GUI performance dashboard.")
        print()
        
        # Setup performance monitoring
        await self.setup_performance_monitoring()
        
        # Start demo
        self.demo_running = True
        
        # Start query simulation in background thread
        query_thread = threading.Thread(
            target=self.simulate_ai_queries,
            daemon=True
        )
        query_thread.start()
        
        # Start display updates in background thread
        display_thread = threading.Thread(
            target=self.display_performance_data,
            daemon=True
        )
        display_thread.start()
        
        print("⏳ Demo starting in 3 seconds...")
        await asyncio.sleep(3)
        
        print("🎯 Demo will run for 10 seconds to test functionality...")
        demo_start_time = time.time()
        demo_duration = 10.0  # Run for 10 seconds
        
        try:
            # Keep demo running for limited time
            while time.time() - demo_start_time < demo_duration:
                await asyncio.sleep(0.5)  # Check more frequently
                
            # Auto-stop after timeout
            print(f"\n⏰ Demo completed after {demo_duration} seconds")
            self.demo_running = False
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo stopped by user...")
            self.demo_running = False
            
            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            print("✅ Demo stopped successfully")
            
            # Show final summary
            if self.performance_monitor:
                summary = self.performance_monitor.get_performance_summary()
                print(f"\n📊 FINAL SUMMARY")
                print(f"   Total Queries Processed: {summary.get('total_requests', 0)}")
                print(f"   Average Response Time: {summary.get('avg_response_time', 0):.0f}ms")
                print(f"   Success Rate: {summary.get('success_rate', 0):.1f}%")


async def main():
    """Main demo function"""
    demo = PerformanceDashboardDemo()
    await demo.run_demo()


if __name__ == "__main__":
    print("🎯 Performance Dashboard Integration Demo")
    print("   Sovereign AI Agent - Task 11.3 Implementation")
    print()
    
    # Run the demo
    asyncio.run(main()) 