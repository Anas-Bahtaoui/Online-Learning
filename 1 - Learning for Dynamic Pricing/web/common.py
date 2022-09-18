from typing import NamedTuple, List, Optional, Dict

from Learner import Reward, PriceIndexes, ProductRewards, Learner, ExperimentHistoryItem
from change_detectors import ChangeHistoryItem
from entities import Product, Customer, Simulation
from parameter_estimators import ParameterHistoryEntry


class IDs(NamedTuple):
    storage: str = "storage"
    left_experiment_selector: str = "experiment-selector-left"
    right_experiment_selector: str = "experiment-selector-right"
    result_div_left: str = "result-div-left"
    result_div_right: str = "result-div-right"
    run_experiment: str = "run-experiment"
    reset_results: str = "reset-results"
    run_count: str = "run-count"
    experiment_count: str = "experiment-count"
    resolution_selector: str = "resolution-selector"
    customer_day_selector: str = "customer-day-selector"
    experiment_aggregate_selector: str = "experiment-aggregate-selector"
    experiment_day_selector: str = "experiment-day-selector"
    experiment_toggle: str = "experiment-toggle"


ids = IDs()


class SimulationResult(NamedTuple):
    """
    Result of one simulation run
    """
    rewards: List[Reward]
    price_indexes: List[PriceIndexes]
    product_rewards: List[ProductRewards]
    products: List[Product]
    customers: Optional[List[List[dict]]]
    estimators: Optional[Dict[str, List[ParameterHistoryEntry]]]
    change_detected_at: List[int]
    change_history: Optional[List[ChangeHistoryItem]]
    clairvoyants: List[float]
    absolute_clairvoyant: float

    def serialize(self):
        result = {
            "rewards": self.rewards,
            "price_indexes": self.price_indexes,
            "product_rewards": self.product_rewards,
            "products": [product.serialize() for product in self.products],
            "change_detected_at": self.change_detected_at,
            "change_history": self.change_history,
            "absolute_clairvoyant": self.absolute_clairvoyant,
            "clairvoyants": self.clairvoyants,
            "estimators": self.estimators,
            "customers": self.customers,
        }
        return result

    @staticmethod
    def deserialize(data):
        return SimulationResult(
            rewards=data["rewards"],
            price_indexes=data["price_indexes"],
            product_rewards=data["product_rewards"],
            products=[Product(*product.values()) for product in data["products"]],
            customers=data.get("customers"),
            estimators=data.get("estimators"),
            change_detected_at=data["change_detected_at"],
            change_history=data["change_history"],
            clairvoyants=data["clairvoyants"],
            absolute_clairvoyant=data["absolute_clairvoyant"],
        )

    @staticmethod
    def from_result(exps: List[ExperimentHistoryItem], products: List[Product], absolute_clairvoyant: float):
        rewards, price_indexes, product_rewards, change_detected_at, change_history, clairvoyants, customers, estimators = zip(
            *exps)
        change_indexes = [ind for ind, value in enumerate(change_detected_at) if value]
        customers, estimators, change_history = list(customers), list(estimators), list(change_history)
        return SimulationResult(
            rewards=list(rewards),
            price_indexes=list(price_indexes),
            product_rewards=list(product_rewards),
            products=products,
            customers=[[customer.serialize() for customer in day] for day in customers] if customers[0] is not None else None,
            estimators=estimators if estimators[0] is not None else None,
            change_detected_at=change_indexes,
            change_history=change_history if change_history[0] is not None else [],
            clairvoyants=list(clairvoyants),
            absolute_clairvoyant=absolute_clairvoyant,
        )
