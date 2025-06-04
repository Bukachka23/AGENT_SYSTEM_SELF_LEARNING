import gc
import sys
from datetime import datetime

from src.config.config import Config
from src.config.logger import Logger
from src.config.settings import LLMConfig
from src.core.agent import SelfImprovingAgent
from src.core.problem_generator import ProblemGenerator
from src.prompts.prompt_manager import PromptManager
from src.utils.resources import memory_monitor


def print_banner(logger):
    """Print startup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║          🤖 Self-Improving AI Agent v2.0 🤖          ║
    ║                                                      ║
    ║  An autonomous agent that learns and improves        ║
    ║  through iterative problem-solving cycles            ║
    ╚══════════════════════════════════════════════════════╝
    """
    logger.info(banner)


def main():
    """Main function with enhanced logging and monitoring."""
    config = Config()
    logger = Logger(name="self_improving_agent", level=logging.INFO)
    llm_config = LLMConfig()

    PromptManager.set_logger(logger)

    try:
        print_banner(logger)

        logger.info(f"📅 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🐍 Python version: {sys.version.split()[0]}")


        if not config.google_api_key:
            logger.error("❌ Google API key not found in environment variables!")
            logger.info("Please set GOOGLE_API_KEY in your .env file")
            return

        logger.info("✅ Configuration loaded successfully")

        with memory_monitor(logger, "Agent Initialization"):
            agent = SelfImprovingAgent(config.google_api_key, logger, llm_config)

        problem_gen = ProblemGenerator()
        logger.info(f"📚 Problem generator ready with {len(problem_gen.problem_templates)} problems")

        logger.info("\n🤖 Self-Improving Agent Demo")
        logger.info("This agent will attempt to solve problems and improve over time")
        logger.info("=" * 60)

        cycles = 3

        problems_per_cycle = 5
        all_problems = list(problem_gen.iterate_problems())

        problems = []
        for cycle in range(cycles):
            for i in range(problems_per_cycle):
                problem_index = i % len(all_problems)
                problems.append(all_problems[problem_index])
        
        logger.info(f"🎯 Generated {len(problems)} problems for {cycles} cycles")

        with memory_monitor(logger, "Improvement Cycles"):
            agent.run_improvement_cycle_generator(iter(problems), cycles)

        gc.collect()
        logger.info("✅ Improvement cycles completed")

        logger.info("\n" + agent.get_performance_report())

        logger.info("\n" + "=" * 50)
        logger.info("🧪 TESTING IMPROVED AGENT")
        logger.info("=" * 50)

        final_problem = "Create an efficient algorithm to sort a large dataset"
        logger.info(f"📝 Final test problem: {final_problem}")

        with memory_monitor(logger, "Final Problem Solving"):
            final_result = agent.solve_problem(final_problem)

        logger.info(
            f"\n✅ Final Problem Solution:"
            f"\n  • Quality Score: {final_result.quality_score:.2f}"
            f"\n  • Execution Time: {final_result.execution_time:.2f}s"
            f"\n  • Approach: {final_result.approach}"
        )

        state_file = f"agent_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.info(f"\n💾 Saving agent state to: {state_file}")
        agent.save_state(state_file)

        logger.info("\n📊 FINAL STATISTICS:")
        logger.info("=" * 50)

        logger.info("Agent Performance:")
        for key, value in agent.stats.items():
            logger.info(f"  • {key.replace('_', ' ').title()}: {value}")

        prompt_stats = PromptManager.get_stats()
        logger.info("\nPrompt Manager:")
        logger.info(f"  • Cache Hit Rate: {prompt_stats['cache_hit_rate']:.1f}%")
        logger.info(f"  • Total Requests: {prompt_stats['total_requests']}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {str(e)}")
        logger.exception("Error details:")
    finally:
        logger.info("\n🧹 Performing final cleanup...")

        if 'agent' in locals():
            agent.cleanup()

        PromptManager.clear_cache()

        collected = gc.collect()
        logger.info(f"🗑️ Final garbage collection: {collected} objects freed")

        logger.info("👋 Agent shutdown complete")
        logger.close()


if __name__ == "__main__":
    import logging
    print("\n" + "=" * 60)
    main()
