import gc
import json
import os
import sys
import time
import weakref
from collections import deque
from contextlib import contextmanager
from typing import Iterator
from typing import Optional, Any

import psutil

from src.config.settings import SolutionResult, PerformanceMetrics
from src.core.interfaces import AIModelInterface, SystemMonitorInterface, StateStorageInterface
from src.infrastructure.models import GeminiAIModel, FileStateStorage
from src.infrastructure.monitoring import PsutilSystemMonitor
from src.infrastructure.storage import MemoryEfficientStorage
from src.prompts.prompt_manager import PromptManager
from src.services.analyzers import ApproachAnalyzer, StrategyAnalyzer
from src.services.categorizers import ProblemCategorizer
from src.services.evaluators import SolutionEvaluator
from src.services.learners import CapabilityLearner


class SelfImprovingAgent:
    """Memory-efficient self-improving agent with comprehensive logging."""

    def __init__(self, api_key: str, logger: Any, llm_config: Any,
                 ai_model: Optional[AIModelInterface] = None,
                 state_storage: Optional[StateStorageInterface] = None,
                 system_monitor: Optional[SystemMonitorInterface] = None):
        # Infrastructure dependencies
        self.logger = logger
        self.ai_model = ai_model or GeminiAIModel(api_key, llm_config.model)
        self.state_storage = state_storage or FileStateStorage()
        self.system_monitor = system_monitor or PsutilSystemMonitor()

        # Domain services
        self.solution_evaluator = SolutionEvaluator()
        self.approach_analyzer = ApproachAnalyzer()
        self.problem_categorizer = ProblemCategorizer()
        self.capability_learner = CapabilityLearner()
        self.strategy_analyzer = StrategyAnalyzer()

        # Original initialization code (unchanged structure)
        self.logger.info("🚀 Initializing Self-Improving Agent")
        self.logger.debug(f"LLM Config: {llm_config}")
        self.api_key = api_key
        self.llm_config = llm_config
        self.logger.info(f"Configuring Gemini AI with model: {llm_config.model}")
        self.model = self.ai_model  # Use injected model
        self.logger.info("✅ Gemini AI configured successfully")

        # Rest of initialization remains the same
        self.logger.info("Setting up memory-efficient storage systems")
        self.performance_history = MemoryEfficientStorage(max_items=100)
        self.solutions_cache = weakref.WeakValueDictionary()
        self.successful_strategies = deque(maxlen=50)
        self.failed_attempts = deque(maxlen=20)
        self._capabilities = {
            'problem_solving': llm_config.problem_solving,
            'code_generation': llm_config.code_generation,
            'optimization': 0.5,
            'error_handling': llm_config.error_handling,
            'efficiency': llm_config.learning_efficiency
        }
        self.logger.info(f"Initial capabilities: {json.dumps(self._capabilities, indent=2)}")
        self.learned_patterns = deque(maxlen=30)
        self.current_cycle = 0
        self.total_problems_solved = 0
        self.memory_monitor_enabled = True
        self.prompt_manager = PromptManager()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_api_calls': 0,
            'total_errors': 0,
            'total_memory_cleanups': 0
        }
        self.logger.info("✅ Agent initialization complete")
        self._log_system_info()

    def _log_system_info(self):
        """Log system information for debugging."""
        cpu_percent = self.system_monitor.get_cpu_percent()
        memory_mb = self.system_monitor.get_process_memory_mb()
        self.logger.debug(f"System Info - CPU: {cpu_percent}%, "
                          f"Memory: {memory_mb:.2f} MB, "
                          f"Python: {sys.version.split()[0]}")

    @contextmanager
    def memory_tracking(self, operation_name: str):
        """Context manager for tracking memory usage during operations."""
        if not self.memory_monitor_enabled:
            yield
            return

        start_memory = self.system_monitor.get_process_memory_mb()
        start_time = time.time()

        self.logger.debug(f"📊 Starting {operation_name} - Memory: {start_memory:.2f} MB")

        try:
            yield
        finally:
            end_memory = self.system_monitor.get_process_memory_mb()
            end_time = time.time()
            memory_delta = end_memory - start_memory
            time_delta = end_time - start_time

            self.logger.info(
                f"📊 Completed {operation_name} - "
                f"Duration: {time_delta:.2f}s, "
                f"Memory Δ: {memory_delta:+.2f} MB"
            )

            self.logger.performance("operation_metrics",
                                    {
                                        "duration": time_delta,
                                        "memory_delta": memory_delta,
                                        "final_memory": end_memory
                                    },
                                    {"operation": operation_name}
                                    )

    def solve_problem(self, problem: str) -> SolutionResult:
        """Solve a single problem with memory-efficient approach."""
        self.logger.debug(f"🔍 Attempting to solve: {problem[:100]}...")

        # Cache check remains the same
        cache_key = hash(problem)
        if cache_key in self.solutions_cache:
            cached_result = self.solutions_cache[cache_key]
            self.logger.info(f"✨ Cache hit for problem: {problem[:50]}...")
            self.stats['cache_hits'] += 1

            self.logger.performance(
                "cache_hit",
                1,
                {"problem": problem[:50], "quality": cached_result.quality_score}
            )
            return cached_result

        self.stats['cache_misses'] += 1
        self.logger.debug("Cache miss - generating new solution")

        start_time = time.time()
        start_memory = self.system_monitor.get_memory_info().rss

        try:
            self.logger.debug("🤖 Calling Gemini AI for solution generation")
            solution_prompt = self.prompt_manager.get_prompt(
                'solution',
                problem=problem,
                capabilities=json.dumps(self._capabilities),
                successful_strategies=json.dumps(list(self.successful_strategies)[-5:])
            )

            self.stats['total_api_calls'] += 1
            api_start = time.time()
            response = self.ai_model.generate_content(solution_prompt)
            api_time = time.time() - api_start

            self.logger.debug(f"✅ Gemini API responded in {api_time:.2f}s")
            solution_text = response.text

            self.logger.debug("📊 Evaluating solution quality")
            quality_score = self._evaluate_solution(problem, solution_text)

            result = SolutionResult(
                problem=problem,
                solution=solution_text,
                quality_score=quality_score,
                execution_time=time.time() - start_time,
                memory_used=(self.system_monitor.get_memory_info().rss - start_memory) / 1024 / 1024,
                approach=self._determine_approach(solution_text)
            )

            self.solutions_cache[cache_key] = result
            self.total_problems_solved += 1

            self.logger.info(
                f"🎯 Solution generated - Quality: {quality_score:.2f}, "
                f"API Time: {api_time:.2f}s, Total Time: {result.execution_time:.2f}s"
            )

            self.logger.performance(
                "problem_solved",
                {
                    "quality": quality_score,
                    "api_time": api_time,
                    "total_time": result.execution_time,
                    "memory_used": result.memory_used
                },
                {"problem": problem[:50], "approach": result.approach}
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Error solving problem: {str(e)}")
            self.logger.exception("Detailed error information:")
            self.stats['total_errors'] += 1

            return SolutionResult(
                problem=problem,
                solution="",
                quality_score=0.0,
                execution_time=time.time() - start_time,
                memory_used=0.0,
                error=str(e)
            )

    def _evaluate_solution(self, problem: str, solution: str) -> float:
        """Evaluate solution quality with minimal memory footprint."""
        self.logger.debug("🔍 Evaluating solution quality")

        try:
            eval_prompt = self.prompt_manager.get_prompt(
                'evaluation',
                problem=problem,
                solution_text=solution[:1000]
            )

            self.stats['total_api_calls'] += 1
            response = self.ai_model.generate_content(eval_prompt)

            rating = self.solution_evaluator.extract_rating_from_response(response.text)
            self.logger.debug(f"📊 Solution evaluated - Score: {rating}")
            return rating

        except Exception as e:
            self.logger.error(f"❌ Error evaluating solution: {str(e)}")
            return 0.5

    def _determine_approach(self, solution: str) -> str:
        """Determine the approach used in the solution."""
        approach = self.approach_analyzer._determine_approach(solution)
        self.logger.debug(f"🏷️ Detected approach: {approach}")
        return approach

    def _categorize_problem(self, problem: str) -> str:
        """Categorize problem type for learning."""
        return self.problem_categorizer._categorize_problem(problem)

    def _learn_from_cycle(self, results: list[SolutionResult]) -> None:
        """Learn from cycle results without keeping all data in memory."""
        if not results:
            self.logger.warning("No results to learn from")
            return

        self.logger.info("🧠 Analyzing cycle results for learning")

        # Use domain service for metrics calculation
        metrics = self.capability_learner.calculate_metrics(results)

        self.logger.info(
            f"📊 Cycle metrics - Avg Score: {metrics['avg_score']:.2f}, "
            f"Success Rate: {metrics['success_rate']:.2%}, "
            f"Successful: {metrics['successful_count']}/{len(results)}"
        )

        # Update capabilities using domain service
        old_capabilities = dict(self._capabilities)
        new_capabilities = self.capability_learner.calculate_capability_improvements(
            self._capabilities, metrics['avg_score']
        )

        for capability, new_value in new_capabilities.items():
            old_value = old_capabilities[capability]
            self._capabilities[capability] = new_value

            if new_value > old_value:
                self.logger.info(
                    f"📈 Capability improved: {capability} "
                    f"{old_value:.3f} → {new_value:.3f} (+{new_value - old_value:.3f})"
                )

        # Extract and store patterns
        if metrics['successful_count'] > 0:
            pattern = {
                'cycle': self.current_cycle,
                'avg_score': metrics['avg_score'],
                'success_rate': metrics['success_rate'],
                'top_approaches': self._get_top_approaches()
            }
            self.learned_patterns.append(pattern)

            self.logger.info(f"🎯 Learned pattern - Top approaches: {pattern['top_approaches']}")

            self.logger.learning(
                "cycle_pattern",
                {
                    "cycle": self.current_cycle,
                    "pattern": pattern,
                    "capabilities": self._capabilities
                }
            )

    def _get_top_approaches(self) -> list[str]:
        """Get top performing approaches from recent strategies."""
        top_approaches = self.strategy_analyzer._get_top_approaches(
            list(self.successful_strategies)
        )
        self.logger.debug(f"Top approaches: {top_approaches}")
        return top_approaches

    def save_state(self, filepath: str) -> None:
        """Save agent state to file for persistence."""
        self.logger.info(f"💾 Saving agent state to {filepath}")

        try:
            state = {
                'capabilities': self._capabilities,
                'total_problems_solved': self.total_problems_solved,
                'current_cycle': self.current_cycle,
                'learned_patterns': list(self.learned_patterns)[-10:],
                'successful_strategies': list(self.successful_strategies)[-20:],
                'stats': self.stats,
                'timestamp': time.time()
            }

            self.state_storage.save(state, filepath)

            self.logger.info(f"✅ State saved successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to save state: {str(e)}")
            self.logger.exception("Save state error details:")

    def load_state(self, filepath: str) -> None:
        """Load agent state from file."""
        self.logger.info(f"📂 Loading agent state from {filepath}")

        try:
            state = self.state_storage.load(filepath)

            if state:
                self._capabilities = state.get('capabilities', self._capabilities)
                self.total_problems_solved = state.get('total_problems_solved', 0)
                self.current_cycle = state.get('current_cycle', 0)
                self.stats = state.get('stats', self.stats)

                patterns = state.get('learned_patterns', [])
                for pattern in patterns:
                    self.learned_patterns.append(pattern)

                strategies = state.get('successful_strategies', [])
                for strategy in strategies:
                    self.successful_strategies.append(strategy)

                saved_time = state.get('timestamp', 0)
                age_hours = (time.time() - saved_time) / 3600

                self.logger.info(
                    f"✅ State loaded successfully - "
                    f"Age: {age_hours:.1f} hours, "
                    f"Cycles: {self.current_cycle}, "
                    f"Problems: {self.total_problems_solved}"
                )

        except Exception as e:
            self.logger.error(f"❌ Error loading state: {str(e)}")
            self.logger.exception("Load state error details:")

    # All other methods remain structurally the same but use the injected services
    def run_improvement_cycle(self, problems: list[str], cycles: int) -> None:
        """Run improvement cycles with list of problems (backward compatibility)."""
        self.logger.info(f"🔄 Starting improvement cycles - Total cycles: {cycles}, Problems: {len(problems)}")
        self.run_improvement_cycle_generator(iter(problems), cycles)

    def run_improvement_cycle_generator(self, problems: Iterator[str], cycles: int) -> None:
        """Run improvement cycles using generator pattern for memory efficiency."""
        self.logger.info(f"🎯 Beginning {cycles} improvement cycles with generator pattern")

        initial_capabilities = dict(self._capabilities)

        for cycle in range(cycles):
            self.current_cycle = cycle + 1
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"📊 IMPROVEMENT CYCLE {self.current_cycle}/{cycles}")
            self.logger.info(f"{'=' * 60}")

            with self.memory_tracking(f"Cycle {self.current_cycle}"):
                cycle_results = []
                cycle_start_time = time.time()
                problems_processed = 0

                for problem_idx, problem in enumerate(problems):
                    problems_processed += 1
                    self.logger.info(f"\n🧩 Problem {problem_idx + 1}: {problem[:50]}...")

                    with self.memory_tracking(f"Problem {problem_idx + 1}"):
                        try:
                            result = self.solve_problem(problem)
                            cycle_results.append(result)

                            self._process_result(result)

                            self.logger.info(
                                f"✅ Problem solved - Quality: {result.quality_score:.2f}, "
                                f"Time: {result.execution_time:.2f}s, "
                                f"Approach: {result.approach}"
                            )

                        except Exception as e:
                            self.logger.error(f"❌ Failed to solve problem: {str(e)}")
                            self.logger.exception("Problem solving error details:")
                            self.stats['total_errors'] += 1

                        if problem_idx % 5 == 0 and problem_idx > 0:
                            self.logger.debug("🧹 Performing periodic cleanup")
                            self._periodic_cleanup()

                self.logger.info(f"\n📚 Learning from {problems_processed} problems...")
                self._learn_from_cycle(cycle_results)

                cycle_time = time.time() - cycle_start_time
                self._record_cycle_performance(cycle_results, cycle_time)

                self._log_cycle_summary(cycle_results, cycle_time, problems_processed)

                cycle_results.clear()
                self._cycle_cleanup()

        self._log_improvement_summary(initial_capabilities)

    def _process_result(self, result: SolutionResult) -> None:
        """Process result immediately to extract learnings."""
        self.logger.debug(f"📝 Processing result - Score: {result.quality_score}")

        if result.quality_score > 0.7:
            strategy = {
                'approach': result.approach,
                'score': result.quality_score,
                'execution_time': result.execution_time
            }
            self.successful_strategies.append(strategy)
            self.logger.debug(f"✅ Added successful strategy: {result.approach}")

            self.logger.learning(
                "successful_strategy",
                {
                    "approach": result.approach,
                    "score": result.quality_score,
                    "problem": result.problem[:50]
                }
            )
        else:
            failure = {
                'problem_type': self._categorize_problem(result.problem),
                'score': result.quality_score,
                'error': result.error
            }
            self.failed_attempts.append(failure)
            self.logger.debug(f"❌ Recorded failed attempt: {failure['problem_type']}")

            self.logger.learning(
                "failed_attempt",
                {
                    "problem_type": failure['problem_type'],
                    "score": result.quality_score,
                    "error": result.error
                }
            )

    def _record_cycle_performance(self, results: list[SolutionResult], cycle_time: float) -> None:
        """Record cycle performance metrics."""
        if not results:
            return

        process = psutil.Process(os.getpid())
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cycle=self.current_cycle,
            avg_score=sum(r.quality_score for r in results) / len(results),
            memory_used=process.memory_info().rss / 1024 / 1024,  # MB
            success_rate=len([r for r in results if r.quality_score > 0.7]) / len(results)
        )

        self.performance_history.add(metrics)

        self.logger.info(
            f"⏱️ Cycle {self.current_cycle} completed in {cycle_time:.2f}s, "
            f"Memory: {metrics.memory_used:.2f} MB"
        )

        self.logger.performance(
            "cycle_complete",
            metrics.to_dict(),
            {"cycle_time": cycle_time, "problems_count": len(results)}
        )

    def _log_cycle_summary(self, results: list[SolutionResult], cycle_time: float, problems_processed: int):
        """Log comprehensive cycle summary."""
        if not results:
            return

        successful = len([r for r in results if r.quality_score > 0.7])
        failed = len([r for r in results if r.quality_score <= 0.7])
        avg_time = sum(r.execution_time for r in results) / len(results)

        summary = f"\n📊 CYCLE {self.current_cycle} SUMMARY:\n"
        summary += f"  • Problems processed: {problems_processed}\n"
        summary += f"  • Successful solutions: {successful} ({successful / len(results) * 100:.1f}%)\n"
        summary += f"  • Failed solutions: {failed}\n"
        summary += f"  • Average solution time: {avg_time:.2f}s\n"
        summary += f"  • Total cycle time: {cycle_time:.2f}s\n"
        summary += f"  • Cache performance: {self.stats['cache_hits']}/{self.stats['cache_hits'] + self.stats['cache_misses']} hits\n"

        self.logger.info(summary)

    def _log_improvement_summary(self, initial_capabilities: dict[str, float]):
        """Log overall improvement summary."""
        self.logger.info("\n🎯 OVERALL IMPROVEMENT SUMMARY:")
        self.logger.info("Capability improvements:")

        total_improvement = 0
        for capability, initial_value in initial_capabilities.items():
            current_value = self._capabilities[capability]
            improvement = current_value - initial_value
            total_improvement += improvement

            symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            self.logger.info(
                f"  {symbol} {capability}: {initial_value:.3f} → {current_value:.3f} "
                f"({improvement:+.3f})"
            )

        avg_improvement = total_improvement / len(initial_capabilities)
        self.logger.info(f"\n  Average improvement: {avg_improvement:+.3f}")
        self.logger.info(f"  Total problems solved: {self.total_problems_solved}")
        self.logger.info(f"  Total API calls: {self.stats['total_api_calls']}")
        self.logger.info(f"  Total errors: {self.stats['total_errors']}")


    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup during processing."""
        self.logger.debug("🧹 Starting periodic cleanup")
        initial_cache_size = len(self.solutions_cache)

        if len(self.solutions_cache) > 50:
            keys_to_remove = list(self.solutions_cache.keys())[:25]
            for key in keys_to_remove:
                if key in self.solutions_cache:
                    del self.solutions_cache[key]

            self.logger.debug(f"🗑️ Removed {len(keys_to_remove)} cache entries")

        if self.total_problems_solved % 20 == 0:
            self.prompt_manager.clear_cache()
            self.logger.debug("🗑️ Cleared prompt cache")

        collected = gc.collect()
        self.stats['total_memory_cleanups'] += 1

        self.logger.debug(
            f"🧹 Cleanup complete - Cache: {initial_cache_size} → {len(self.solutions_cache)}, "
            f"GC collected: {collected} objects"
        )

    def _cycle_cleanup(self) -> None:
        """Cleanup after each cycle."""
        self.logger.debug("🧹 Performing cycle cleanup")

        # Trim collections to ensure they stay within bounds
        strategies_before = len(self.successful_strategies)
        attempts_before = len(self.failed_attempts)

        while len(self.successful_strategies) > 50:
            self.successful_strategies.popleft()

        while len(self.failed_attempts) > 20:
            self.failed_attempts.popleft()

        # Clear some cache
        self._periodic_cleanup()

        # Log memory status
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        self.logger.info(
            f"🧹 Cycle cleanup complete - "
            f"Memory: {memory_mb:.2f} MB, "
            f"Strategies: {strategies_before} → {len(self.successful_strategies)}, "
            f"Failed: {attempts_before} → {len(self.failed_attempts)}"
        )

    def get_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        self.logger.info("📋 Generating performance report")

        report = "📈 PERFORMANCE REPORT\n"
        report += "=" * 50 + "\n\n"

        report += "📊 Overall Statistics:\n"
        report += f"  • Total problems solved: {self.total_problems_solved}\n"
        report += f"  • Total cycles completed: {self.current_cycle}\n"
        report += f"  • Successful strategies: {len(self.successful_strategies)}\n"
        report += f"  • Failed attempts: {len(self.failed_attempts)}\n"
        report += f"  • Cache hit rate: {self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'] + 0.001) * 100:.1f}%\n"
        report += f"  • Total API calls: {self.stats['total_api_calls']}\n"
        report += f"  • Total errors: {self.stats['total_errors']}\n"
        report += f"  • Memory cleanups: {self.stats['total_memory_cleanups']}\n\n"

        report += "🎯 Current Capabilities:\n"
        for capability, score in sorted(self._capabilities.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            report += f"  {capability:20} [{bar}] {score:.2f}\n"
        report += "\n"

        recent_metrics = self.performance_history.get_recent(5)
        if recent_metrics:
            report += "📈 Recent Performance:\n"
            for metric in recent_metrics:
                report += (
                    f"  Cycle {metric.cycle}: "
                    f"Score={metric.avg_score:.2f}, "
                    f"Success={metric.success_rate:.1%}, "
                    f"Memory={metric.memory_used:.1f}MB\n"
                )
            report += "\n"

        # Top approaches
        top_approaches = self._get_top_approaches()
        if top_approaches:
            report += f"🏆 Top Performing Approaches: {', '.join(top_approaches)}\n\n"

        # Memory usage
        process = psutil.Process(os.getpid())
        report += f"💾 Current Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB\n"

        return report

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.logger.info("🧹 Starting agent cleanup")

        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"  • Total problems solved: {self.total_problems_solved}")
        self.logger.info(
            f"  • Cache hit rate: {self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'] + 0.001) * 100:.1f}%")
        self.logger.info(f"  • Total API calls: {self.stats['total_api_calls']}")
        self.logger.info(f"  • Total errors: {self.stats['total_errors']}")

        self.performance_history.clear()
        self.solutions_cache.clear()
        self.successful_strategies.clear()
        self.failed_attempts.clear()
        self.learned_patterns.clear()

        self.prompt_manager.clear_cache()

        collected = gc.collect()

        process = psutil.Process(os.getpid())
        final_memory = process.memory_info().rss / 1024 / 1024

        self.logger.info(
            f"✅ Cleanup complete - "
            f"Final memory: {final_memory:.2f} MB, "
            f"GC collected: {collected} objects"
        )
