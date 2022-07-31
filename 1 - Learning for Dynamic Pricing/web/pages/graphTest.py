import dash
from graphs import render_for_learner
from db_cache import get_cache
from production import config, learners, RUN_COUNT
from common import SimulationResult

dash.register_page(__name__, order=5)
cache_key = (*(learner.name for learner in learners), *["Production" for _ in range(4)], RUN_COUNT)
production_results = get_cache(cache_key)
greedy = SimulationResult.deserialize(production_results["Greedy Algorithm"])
layout = [
    render_for_learner("Greedy Learner", greedy)
]
