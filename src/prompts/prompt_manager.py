import gc
import json
import os
import time
from typing import Dict, Any


class PromptManager:
    """Manages prompt loading and caching with memory efficiency and logging."""

    _cache = {}
    _max_cache_size = 100
    _prompt_templates = {
        'analysis': 'analysis.txt',
        'evaluation': 'evaluation.txt',
        'improvement': 'improvement.txt',
        'learning': 'learning.txt',
        'solution': 'solution.txt'
    }
    _stats = {
        'cache_hits': 0,
        'cache_misses': 0,
        'file_loads': 0,
        'fallback_used': 0,
        'format_errors': 0,
        'cache_evictions': 0
    }

    _logger = None

    @classmethod
    def set_logger(cls, logger):
        """Set logger for the class."""
        cls._logger = logger
        if logger:
            logger.debug("PromptManager logger configured")

    @classmethod
    def _manage_cache_size(cls):
        """Manage cache size to prevent memory issues."""
        if len(cls._cache) >= cls._max_cache_size:
            cache_items = list(cls._cache.items())
            items_to_remove = len(cache_items) // 2
            
            for i in range(items_to_remove):
                key, _ = cache_items[i]
                del cls._cache[key]
            
            cls._stats['cache_evictions'] += items_to_remove
            
            if cls._logger:
                cls._logger.debug(f"🗑️ Cache size management: removed {items_to_remove} entries")

    @classmethod
    def get_prompt(cls, prompt_type: str, **kwargs) -> str:
        """Load and format prompt with lazy loading."""
        start_time = time.time()
        cache_key = f"{prompt_type}_{hash(frozenset(kwargs.items()))}"

        if cls._logger:
            cls._logger.debug(f"📄 Requesting prompt: {prompt_type}")

        if cache_key in cls._cache:
            cls._stats['cache_hits'] += 1
            if cls._logger:
                cls._logger.debug(f"✨ Prompt cache hit for: {prompt_type}")
            return cls._cache[cache_key]

        cls._stats['cache_misses'] += 1
        if cls._logger:
            cls._logger.debug(f"Cache miss for prompt: {prompt_type}")

        try:
            template_file = cls._prompt_templates.get(prompt_type)
            prompt_path = os.path.join(os.path.dirname(__file__), template_file)
            if template_file and os.path.exists(prompt_path):
                if cls._logger:
                    cls._logger.debug(f"📂 Loading prompt from file: {template_file}")

                with open(prompt_path, 'r') as f:
                    template = f.read()
                cls._stats['file_loads'] += 1
            else:
                if cls._logger:
                    cls._logger.warning(
                        f"⚠️ Prompt file not found: {template_file}, using inline template"
                    )
                template = cls._get_inline_template(prompt_type)

            try:
                formatted = template.format(**kwargs)
            except KeyError as e:
                cls._stats['format_errors'] += 1
                if cls._logger:
                    cls._logger.error(f"❌ Error formatting prompt: Missing key {e}")
                return cls._get_fallback_template(prompt_type, **kwargs)

            cls._manage_cache_size()
            cls._cache[cache_key] = formatted

            load_time = time.time() - start_time
            if cls._logger:
                cls._logger.debug(
                    f"✅ Prompt loaded and formatted - "
                    f"Type: {prompt_type}, Time: {load_time:.3f}s, "
                    f"Size: {len(formatted)} chars"
                )

            return formatted

        except Exception as e:
            cls._stats['fallback_used'] += 1
            if cls._logger:
                cls._logger.error(f"❌ Error loading prompt: {str(e)}")
                cls._logger.exception("Prompt loading error details:")
            return cls._get_fallback_template(prompt_type, **kwargs)

    @staticmethod
    def _get_inline_template(prompt_type: str) -> str:
        """Get inline template as fallback."""
        templates = {
            'analysis': "Analyze the task: {task}\nProvide complexity, skills, challenges, approach, and success criteria.",
            'evaluation': "Evaluate the solution: {solution_text} for problem: {problem}\nRate from 0.0 to 1.0",
            'improvement': "Improve this code: {current_code}\nGoal: {improvement_goal}\nCapabilities: {capabilities}",
            'learning': "Analyze performance: {performance_metrics}\nSuccessful: {successful_count}, Failed: {failed_count}",
            'solution': "Solve this problem: {problem}\nUsing capabilities: {capabilities}\nStrategies: {successful_strategies}"
        }
        return templates.get(prompt_type, "Process: {input}")

    @staticmethod
    def _get_fallback_template(prompt_type: str, **kwargs) -> str:
        """Ultimate fallback template."""
        return f"Process {prompt_type} with inputs: {json.dumps(kwargs, indent=2)}"

    @classmethod
    def clear_cache(cls) -> None:
        """Clear prompt cache to free memory."""
        cache_size = len(cls._cache)
        cls._cache.clear()
        collected = gc.collect()

        if cls._logger:
            cls._logger.info(
                f"🗑️ Prompt cache cleared - "
                f"Entries removed: {cache_size}, "
                f"GC collected: {collected} objects"
            )

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get prompt manager statistics."""
        total_requests = cls._stats['cache_hits'] + cls._stats['cache_misses']
        hit_rate = cls._stats['cache_hits'] / max(total_requests, 1) * 100

        stats = {
            **cls._stats,
            'total_requests': total_requests,
            'cache_hit_rate': hit_rate,
            'current_cache_size': len(cls._cache)
        }

        if cls._logger:
            cls._logger.debug(f"Prompt stats: {stats}")

        return stats
