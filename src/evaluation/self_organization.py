"""
Self-organization metrics for traffic systems.

Provides quantitative measures for emergent behavior, coordination,
and self-organizing properties of traffic control systems.

Metrics implemented:
- Green wave index: Measures signal coordination along corridors
- Flow entropy: Measures traffic distribution balance
- Adaptation rate: Measures recovery speed from perturbations
- Local-global correlation: Measures emergence of global patterns

References:
    [1] Kauffman, S. A. (1993). The Origins of Order: Self-Organization
        and Selection in Evolution. Oxford University Press.
    [2] Gershenson, C., & Heylighen, F. (2005). When can we call a system
        self-organizing? Advances in Artificial Life, 606-614.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.state import SimulationState, MetricsSnapshot
from src.entities.network import RoadNetwork
from src.entities.traffic_light import Direction

class SelfOrganizationAnalyzer:
    """
    Compute metrics for self-organizing behavior in traffic systems.

    This class provides tools to quantify emergent properties of the
    traffic control system, including signal coordination, flow balance,
    and adaptation to perturbations.
    """

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._phase_cache: Dict[int, List[float]] = {}

    def compute_green_wave_index(self, state: SimulationState) -> float:
        """
        Measure coordination of signal phases along corridors.

        A "green wave" occurs when consecutive intersections have offset
        phases that allow continuous flow. This metric measures how well
        signals are coordinated.

        Args:
            state: Current simulation state

        Returns:
            Score from 0.0 (no coordination) to 1.0 (perfect green wave)
        """
        if not state.network:
            return 0.0

        coordinated = 0
        total = 0

        for road in state.network.roads.values():
            src_int = state.network.intersections.get(road.from_intersection)
            dst_int = state.network.intersections.get(road.to_intersection)

            if src_int is None or dst_int is None:
                continue

            total += 1

            if road.speed_limit > 0:
                travel_time = road.length / road.speed_limit
            else:
                travel_time = 10.0

            phase_offset = self._compute_phase_offset(src_int, dst_int)

            tolerance = 5.0
            if abs(phase_offset - travel_time) < tolerance:
                coordinated += 1

        return coordinated / max(1, total)

    def _compute_phase_offset(self, src_int, dst_int) -> float:
        """
        Compute the effective phase offset between two intersections.

        Returns the time offset between when src turns green and when
        dst turns green for the connecting direction.
        """
        src_tl = src_int.traffic_light
        dst_tl = dst_int.traffic_light

        src_time = src_tl.get_time_in_phase()
        dst_time = dst_tl.get_time_in_phase()

        offset = dst_time - src_time

        phase_duration = src_tl.current_phase.duration
        if offset < 0:
            offset += phase_duration

        return float(offset)

    def compute_flow_entropy(self, state: SimulationState) -> float:
        """
        Compute Shannon entropy of traffic distribution across roads.

        Low entropy = traffic concentrated on few roads (potential congestion)
        High entropy = traffic evenly distributed (healthy flow)

        Args:
            state: Current simulation state

        Returns:
            Normalized entropy from 0.0 to 1.0
        """
        if not state.network:
            return 0.0

        densities = []
        for road in state.network.roads.values():
            density = road.get_density()
            if density > 0:
                densities.append(density)

        if not densities:
            return 0.0

        total = sum(densities)
        probs = [d / total for d in densities]

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)

        max_entropy = np.log2(len(densities)) if len(densities) > 1 else 1.0

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def compute_spatial_correlation(self, state: SimulationState) -> float:
        """
        Compute spatial autocorrelation of queue lengths.

        High correlation = local patterns, self-organized clusters
        Low correlation = random distribution

        Uses Moran's I statistic adapted for network topology.

        Args:
            state: Current simulation state

        Returns:
            Correlation coefficient from -1.0 to 1.0
        """
        if not state.network:
            return 0.0

        intersections = list(state.network.intersections.values())
        if len(intersections) < 2:
            return 0.0

        queues = [i.get_total_queue() for i in intersections]
        mean_queue = np.mean(queues)

        if np.std(queues) < 0.01:
            return 0.0

        n = len(intersections)
        int_ids = [i.id for i in intersections]
        id_to_idx = {id_: idx for idx, id_ in enumerate(int_ids)}

        numerator = 0.0
        weight_sum = 0.0

        for road in state.network.roads.values():
            i = id_to_idx.get(road.from_intersection)
            j = id_to_idx.get(road.to_intersection)

            if i is None or j is None:
                continue

            weight_sum += 1
            numerator += (queues[i] - mean_queue) * (queues[j] - mean_queue)

        if weight_sum == 0:
            return 0.0

        denominator = sum((q - mean_queue) ** 2 for q in queues)
        if denominator == 0:
            return 0.0

        morans_i = (n / weight_sum) * (numerator / denominator)

        return float(max(-1.0, min(1.0, morans_i)))

    def compute_adaptation_rate(
        self,
        metrics_before: List[MetricsSnapshot],
        metrics_after: List[MetricsSnapshot],
        metric_name: str = "average_delay",
        recovery_threshold: float = 0.9,
    ) -> float:
        """
        Measure how quickly the system recovers from a perturbation.

        Args:
            metrics_before: Metrics snapshots before the perturbation
            metrics_after: Metrics snapshots after the perturbation
            metric_name: Which metric to analyze
            recovery_threshold: Fraction of baseline to consider "recovered"

        Returns:
            Number of ticks to recover (lower is better adaptive)
        """
        if not metrics_before or not metrics_after:
            return float(len(metrics_after)) if metrics_after else 0.0

        baseline_values = [getattr(m, metric_name, 0) for m in metrics_before]
        baseline = np.mean(baseline_values) if baseline_values else 0.0

        if baseline == 0:
            return 0.0

        after_values = [getattr(m, metric_name, 0) for m in metrics_after]

        for i, value in enumerate(after_values):

            if metric_name in ["average_delay", "average_queue_length"]:

                if value <= baseline / recovery_threshold:
                    return float(i)
            else:

                if value >= baseline * recovery_threshold:
                    return float(i)

        return float(len(metrics_after))

    def compute_local_global_correlation(
        self,
        local_metrics: Dict[int, float],
        global_metric: float,
    ) -> float:
        """
        Compute correlation between local decisions and global outcome.

        This measures the degree of emergence in the system:
        - High correlation: Local decisions predict global outcome (not emergent)
        - Low correlation: Global behavior emerges from local interactions

        Args:
            local_metrics: Per-intersection metrics (e.g., queue lengths)
            global_metric: System-wide metric (e.g., total throughput)

        Returns:
            Emergence score from 0.0 (predictable) to 1.0 (emergent)
        """
        if not local_metrics:
            return 0.0

        local_values = list(local_metrics.values())
        local_mean = np.mean(local_values)

        if global_metric == 0:
            return 0.0

        prediction_error = abs(global_metric - local_mean) / abs(global_metric)

        emergence_score = min(1.0, prediction_error)
        return float(emergence_score)

    def compute_phase_synchronization(self, state: SimulationState) -> float:
        """
        Measure synchronization of signal phases across the network.

        Uses Kuramoto order parameter to measure phase coherence.

        Args:
            state: Current simulation state

        Returns:
            Order parameter from 0.0 (incoherent) to 1.0 (synchronized)
        """
        if not state.network:
            return 0.0

        phases = []
        for intersection in state.network.intersections.values():
            tl = intersection.traffic_light

            phase_idx = tl.current_phase_idx
            n_phases = len(tl.phases)
            time_in_phase = tl.get_time_in_phase()
            phase_duration = tl.current_phase.duration

            phase_fraction = time_in_phase / max(1, phase_duration)
            angle = (phase_idx + phase_fraction) * 2 * np.pi / max(1, n_phases)
            phases.append(angle)

        if not phases:
            return 0.0

        complex_phases = np.exp(1j * np.array(phases))
        order_parameter = np.abs(np.mean(complex_phases))

        return float(order_parameter)

    def compute_all_metrics(
        self, state: SimulationState
    ) -> Dict[str, float]:
        """
        Compute all self-organization metrics.

        Args:
            state: Current simulation state

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            "green_wave_index": self.compute_green_wave_index(state),
            "flow_entropy": self.compute_flow_entropy(state),
            "spatial_correlation": self.compute_spatial_correlation(state),
            "phase_synchronization": self.compute_phase_synchronization(state),
        }

        if state.network:
            local_queues = {
                int_id: intersection.get_total_queue()
                for int_id, intersection in state.network.intersections.items()
            }
            global_queue = state.current_metrics.average_queue_length
            metrics["local_global_correlation"] = self.compute_local_global_correlation(
                local_queues, global_queue
            )

        return metrics

def analyze_experiment_self_organization(
    snapshots: List[MetricsSnapshot],
    network: Optional[RoadNetwork] = None,
) -> Dict[str, float]:
    """
    Analyze self-organization metrics over an experiment.

    Computes time-averaged metrics from a sequence of snapshots.

    Args:
        snapshots: List of metrics snapshots from experiment
        network: Optional network for additional analysis

    Returns:
        Dictionary of aggregated metrics
    """
    if not snapshots:
        return {}

    results = {
        "total_ticks": len(snapshots),
    }

    delays = [s.average_delay for s in snapshots]
    queues = [s.average_queue_length for s in snapshots]
    speeds = [s.average_speed for s in snapshots]

    results["delay_mean"] = float(np.mean(delays))
    results["delay_std"] = float(np.std(delays))
    results["delay_trend"] = float(np.polyfit(range(len(delays)), delays, 1)[0])

    results["queue_mean"] = float(np.mean(queues))
    results["queue_std"] = float(np.std(queues))

    results["speed_mean"] = float(np.mean(speeds))
    results["speed_std"] = float(np.std(speeds))

    if np.mean(delays) > 0:
        cv = np.std(delays) / np.mean(delays)
        results["stability"] = float(1.0 / (1.0 + cv))
    else:
        results["stability"] = 1.0

    return results
