# mms_mp/formation_detection.py
"""
Automatic MMS Formation Detection and Analysis
==============================================

This module automatically detects the type of MMS spacecraft formation
(tetrahedral, string-of-pearls, planar, etc.) from actual position data
without making assumptions about the configuration.

Key Features:
- Automatic formation type detection
- Robust geometric analysis
- Adaptive analysis methods based on detected formation
- No assumptions about formation type
"""

import numpy as np
import warnings
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

SQRT2 = np.sqrt(2.0)
REG_TETRA_V_OVER_A3 = 1.0 / (6.0 * SQRT2)  # V/a^3 for regular tetrahedron ≈ 0.11785


class FormationType(Enum):
    """Enumeration of possible MMS formation types"""
    TETRAHEDRAL = "tetrahedral"
    STRING_OF_PEARLS = "string_of_pearls"
    PLANAR = "planar"
    LINEAR = "linear"
    IRREGULAR = "irregular"
    COLLAPSED = "collapsed"

# Classification thresholds (module-level, tweakable)
STRING_LINEARITY_THR = 0.85
STRING_SPHERICITY_MAX = 0.10
STRING_LINEARITY_FALLBACK = 0.75
STRING_SPHERICITY_FALLBACK = 0.20
STRING_AXIS_ANGLE_MAX_DEG = 12.0  # average angle to principal axis for strong string


@dataclass
class FormationAnalysis:
    """Results of formation detection and analysis"""
    formation_type: FormationType
    confidence: float  # 0-1 confidence in detection
    volume: float  # Formation volume in km³
    characteristic_scale: float  # Characteristic size in km
    linearity: float  # 0-1, how linear the formation is
    planarity: float  # 0-1, how planar the formation is
    sphericity: float  # 0-1, how spherical/tetrahedral the formation is
    principal_components: np.ndarray  # Eigenvalues of position covariance
    principal_directions: np.ndarray  # Eigenvectors of position covariance
    spacecraft_ordering: Dict[str, List[str]]  # Ordering in different coordinate systems
    separations: Dict[str, float]  # Inter-spacecraft distances
    formation_center: np.ndarray  # Center of formation
    quality_metrics: Dict[str, float]  # Various quality metrics


def detect_formation_type(positions: Dict[str, np.ndarray],
                         velocities: Optional[Dict[str, np.ndarray]] = None) -> FormationAnalysis:
    """
    Automatically detect MMS formation type from spacecraft positions and velocities

    Parameters:
    -----------
    positions : Dict[str, np.ndarray]
        Spacecraft positions {probe: position_vector} in km
    velocities : Optional[Dict[str, np.ndarray]]
        Spacecraft velocities {probe: velocity_vector} in km/s
        If provided, enables velocity-aware formation analysis

    Returns:
    --------
    FormationAnalysis
        Complete analysis of the detected formation
    """

    # Convert to array for easier manipulation
    probes = ['1', '2', '3', '4']
    pos_array = np.array([positions[probe] for probe in probes])

    # Calculate formation center and centered positions
    formation_center = np.mean(pos_array, axis=0)
    centered_positions = pos_array - formation_center

    # Principal component analysis
    cov_matrix = np.cov(centered_positions.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalue magnitude (largest first)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Ensure eigenvalues are positive (numerical stability)
    eigenvalues = np.abs(eigenvalues)

    # Calculate geometric metrics
    total_variance = np.sum(eigenvalues)
    if total_variance > 0:
        linearity = eigenvalues[0] / total_variance
        planarity = (eigenvalues[0] + eigenvalues[1]) / total_variance
        sphericity = eigenvalues[2] / total_variance if eigenvalues[0] > 0 else 0
    else:
        linearity = planarity = sphericity = 0

    # Calculate formation volume
    if len(pos_array) >= 4:
        # Tetrahedron volume
        matrix = np.array([
            pos_array[1] - pos_array[0],
            pos_array[2] - pos_array[0],
            pos_array[3] - pos_array[0]
        ])
        volume = abs(np.linalg.det(matrix)) / 6.0
    else:
        volume = 0

    # Calculate characteristic scale
    characteristic_scale = np.sqrt(eigenvalues[0]) if eigenvalues[0] > 0 else 0

    # Calculate inter-spacecraft separations
    separations = {}
    for i, probe1 in enumerate(probes):
        for j, probe2 in enumerate(probes):
            if i < j:
                sep = np.linalg.norm(positions[probe1] - positions[probe2])
                separations[f"{probe1}-{probe2}"] = sep

    # Detect formation type based on geometric properties
    formation_type, confidence = _classify_formation(
        linearity, planarity, sphericity, volume, separations, eigenvalues,
        centered_positions=centered_positions, eigenvectors=eigenvectors
    )

    # Calculate spacecraft orderings in different coordinate systems
    spacecraft_ordering = _calculate_orderings(positions, eigenvectors, velocities)

    # Calculate quality metrics
    quality_metrics = _calculate_quality_metrics(
        positions, eigenvalues, separations, formation_type
    )

    return FormationAnalysis(
        formation_type=formation_type,
        confidence=confidence,
        volume=volume,
        characteristic_scale=characteristic_scale,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        principal_components=eigenvalues,
        principal_directions=eigenvectors,
        spacecraft_ordering=spacecraft_ordering,
        separations=separations,
        formation_center=formation_center,
        quality_metrics=quality_metrics
    )


def _classify_formation(linearity: float, planarity: float, sphericity: float,
                       volume: float, separations: Dict[str, float],
                       eigenvalues: np.ndarray,
                       *,
                       centered_positions: Optional[np.ndarray] = None,
                       eigenvectors: Optional[np.ndarray] = None) -> Tuple[FormationType, float]:
    """
    Classify formation type based on geometric properties

    Returns:
    --------
    Tuple[FormationType, float]
        Formation type and confidence (0-1)
    """

    # Check for collapsed formation (all spacecraft very close)
    max_separation = max(separations.values()) if separations else 0
    if max_separation < 10:  # Less than 10 km
        return FormationType.COLLAPSED, 0.9

    # Check for string-of-pearls (high linearity, low sphericity) with axis-angle reinforcement
    if linearity > STRING_LINEARITY_THR and sphericity < STRING_SPHERICITY_MAX:
        axis_ok = True
        axis_angle_mean = None
        if centered_positions is not None and eigenvectors is not None:
            # principal axis is eigenvectors[:,0]
            axis = eigenvectors[:, 0]
            axis = axis / (np.linalg.norm(axis) or 1.0)
            angs = []
            for v in centered_positions:
                if np.linalg.norm(v) == 0:
                    continue
                u = v / np.linalg.norm(v)
                c = np.clip(np.dot(u, axis), -1.0, 1.0)
                angs.append(np.degrees(np.arccos(abs(c))))  # use absolute to ignore direction
            if angs:
                axis_angle_mean = float(np.mean(angs))
                axis_ok = axis_angle_mean <= STRING_AXIS_ANGLE_MAX_DEG
        confidence = min(linearity, 1.0 - sphericity)
        if axis_angle_mean is not None:
            # incorporate alignment (smaller mean angle → higher confidence)
            confidence = float(np.clip(confidence * max(0.5, (STRING_AXIS_ANGLE_MAX_DEG - axis_angle_mean) / STRING_AXIS_ANGLE_MAX_DEG + 0.5), 0.0, 1.0))
        return FormationType.STRING_OF_PEARLS, confidence

    # Check for linear formation (next tier)
    if linearity > STRING_LINEARITY_FALLBACK and sphericity < STRING_SPHERICITY_FALLBACK:
        confidence = float(np.clip(linearity * (1.0 - sphericity), 0.0, 1.0))
        return FormationType.LINEAR, confidence

    # Check for planar formation (high planarity, low sphericity)
    if planarity > 0.9 and sphericity < 0.15:
        confidence = planarity * (1.0 - sphericity)
        return FormationType.PLANAR, confidence

    # Check for tetrahedral formation (balanced eigenvalues, significant volume)
    if sphericity > 0.15 and volume > 1000:  # Significant 3D structure
        # Good tetrahedron has relatively balanced eigenvalues
        if eigenvalues[0] > 0:
            balance = eigenvalues[2] / eigenvalues[0]  # Ratio of smallest to largest
            if balance > 0.1:  # Not too elongated
                confidence = sphericity * balance
                return FormationType.TETRAHEDRAL, confidence

    # Default to irregular if no clear pattern
    confidence = float(np.clip(1.0 - max(linearity, planarity, sphericity), 0.0, 1.0))
    return FormationType.IRREGULAR, confidence


def _calculate_orderings(positions: Dict[str, np.ndarray],
                        eigenvectors: np.ndarray,
                        velocities: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, List[str]]:
    """Calculate spacecraft orderings in different coordinate systems"""

    probes = ['1', '2', '3', '4']
    orderings = {}

    # GSM coordinate orderings
    orderings['X_GSM'] = sorted(probes, key=lambda p: positions[p][0])
    orderings['Y_GSM'] = sorted(probes, key=lambda p: positions[p][1])
    orderings['Z_GSM'] = sorted(probes, key=lambda p: positions[p][2])

    # Radial ordering (distance from Earth)
    orderings['Radial'] = sorted(probes, key=lambda p: np.linalg.norm(positions[p]))

    # Principal component orderings
    formation_center = np.mean([positions[p] for p in probes], axis=0)
    for i, direction in enumerate(['PC1', 'PC2', 'PC3']):
        if i < len(eigenvectors):
            projections = {p: np.dot(positions[p] - formation_center, eigenvectors[:, i])
                          for p in probes}
            orderings[direction] = sorted(probes, key=lambda p: projections[p])

    # Velocity-aware orderings (if velocities provided)
    if velocities is not None:
        # Calculate mean velocity direction
        mean_velocity = np.mean([velocities[p] for p in probes], axis=0)
        velocity_direction = mean_velocity / (np.linalg.norm(mean_velocity) + 1e-12)

        # Order by projection along velocity direction (orbital ordering)
        velocity_projections = {p: np.dot(positions[p] - formation_center, velocity_direction)
                               for p in probes}
        orderings['Orbital'] = sorted(probes, key=lambda p: velocity_projections[p])

        # For string-of-pearls: order by who is "ahead" in orbital motion
        # This is the key insight - spacecraft ahead in orbit are "leading"
        orderings['Leading_to_Trailing'] = sorted(probes, key=lambda p: velocity_projections[p], reverse=True)

        # Cross-track ordering (perpendicular to velocity)
        # Calculate cross-track direction (perpendicular to both velocity and radial)
        radial_direction = formation_center / np.linalg.norm(formation_center)
        cross_track = np.cross(velocity_direction, radial_direction)
        cross_track = cross_track / np.linalg.norm(cross_track)

        cross_projections = {p: np.dot(positions[p] - formation_center, cross_track)
                            for p in probes}
        orderings['Cross_Track'] = sorted(probes, key=lambda p: cross_projections[p])

        # Along-track ordering (along velocity direction)
        orderings['Along_Track'] = orderings['Orbital']  # Same as orbital

    return orderings


def _calculate_quality_metrics(positions: Dict[str, np.ndarray],
                              eigenvalues: np.ndarray,
                              separations: Dict[str, float],
                              formation_type: FormationType) -> Dict[str, float]:
    """Calculate formation-specific quality metrics"""

    metrics = {}

    # Basic metrics
    metrics['mean_separation'] = float(np.mean(list(separations.values())))
    metrics['separation_std'] = float(np.std(list(separations.values())))
    metrics['separation_uniformity'] = float(1.0 - (metrics['separation_std'] / (metrics['mean_separation'] + 1e-12)))

    # Eigenvalue ratios
    # Tetrahedron quality factor (Q) normalized to regular tetrahedron
    # Following the idea: Q = 6*sqrt(2)*V / (sum of edge lengths)^3 scaled proxy
    try:
        # Build full set of six edges for 4 spacecraft
        edges = []
        probes = ['1','2','3','4']
        for i, p1 in enumerate(probes):
            for j, p2 in enumerate(probes):
                if i < j:
                    edges.append(np.linalg.norm(positions[p1] - positions[p2]))
        edges = np.array(edges)
        mean_edge = edges.mean() if len(edges) > 0 else 0.0
        # Approximate regular tetrahedron edge length a by mean edge
        # Regular tetrahedron volume V_reg = REG_TETRA_V_OVER_A3 * a^3
        v_reg = REG_TETRA_V_OVER_A3 * (mean_edge ** 3)
        q_tet = float(min(1.0, (1e-12 + v_reg) and (eigenvalues[0] >= 0) and (eigenvalues[1] >= 0) and (eigenvalues[2] >= 0)))
        # Use ratio of actual volume to regular volume at mean edge as a proxy
        # Clamp to [0,1]
        q_tet = float(np.clip((1e-12 + metrics.get('tetrahedral_quality', 0)) * (volume / (v_reg + 1e-12)), 0.0, 1.0))
        metrics['tetrahedron_quality_factor'] = q_tet
        metrics['mean_edge_km'] = mean_edge
    except Exception:
        pass

    # Separation matrix condition number (uniformity of edges)
    try:
        # Construct separation matrix (pairwise distances), compute std/mean
        if len(edges) > 0:
            sep_std = float(np.std(edges))
            sep_mean = float(np.mean(edges))
            metrics['separation_cv'] = sep_std / (sep_mean + 1e-12)
    except Exception:
        pass

    if eigenvalues[0] > 0:
        metrics['eigenvalue_ratio_21'] = eigenvalues[1] / eigenvalues[0]
        metrics['eigenvalue_ratio_31'] = eigenvalues[2] / eigenvalues[0]
        metrics['eigenvalue_ratio_32'] = eigenvalues[2] / eigenvalues[1] if eigenvalues[1] > 0 else 0

    # Formation-specific quality
    if formation_type == FormationType.STRING_OF_PEARLS:
        # For string-of-pearls, good quality means high linearity and uniform spacing
        metrics['string_quality'] = metrics['separation_uniformity']
    elif formation_type == FormationType.TETRAHEDRAL:
        # For tetrahedral, good quality means balanced eigenvalues
        metrics['tetrahedral_quality'] = metrics.get('eigenvalue_ratio_31', 0)
    elif formation_type == FormationType.PLANAR:
        # For planar, good quality means small out-of-plane component
        metrics['planar_quality'] = 1.0 - metrics.get('eigenvalue_ratio_31', 1)

    return metrics


def get_formation_specific_analysis_method(formation_analysis: FormationAnalysis) -> str:
    """
    Recommend the best analysis method based on detected formation type

    Returns:
    --------
    str
        Recommended analysis method
    """

    formation_type = formation_analysis.formation_type

    if formation_type == FormationType.STRING_OF_PEARLS:
        return "string_of_pearls_timing"
    elif formation_type == FormationType.LINEAR:
        return "linear_timing"
    elif formation_type == FormationType.TETRAHEDRAL:
        return "tetrahedral_timing"
    elif formation_type == FormationType.PLANAR:
        return "planar_timing"
    else:
        return "general_timing"


def analyze_formation_from_event_data(event_data: Dict, target_time,
                                    use_mec_ephemeris: bool = True) -> FormationAnalysis:
    """
    Analyze MMS formation using event data with MEC ephemeris as primary source

    This function ensures that formation analysis always uses MEC ephemeris data
    as the authoritative source for spacecraft positions and velocities.

    Parameters:
    ...
    # After computing velocities and analysis below, record velocity direction if available

    -----------
    event_data : Dict
        Event data from mms_mp.data_loader.load_event()
    target_time : datetime
        Target time for formation analysis
    use_mec_ephemeris : bool
        Whether to use MEC ephemeris (should always be True)

    Returns:
    --------
    FormationAnalysis
        Complete formation analysis using authoritative MEC data
    """

    if not use_mec_ephemeris:
        warnings.warn("use_mec_ephemeris=False is deprecated. MEC data should always be used.")

    # Use ephemeris manager to get authoritative positions and velocities
    from .ephemeris import get_mec_ephemeris_manager

    ephemeris_mgr = get_mec_ephemeris_manager(event_data)
    formation_data = ephemeris_mgr.get_formation_analysis_data(target_time)

    # Perform formation analysis using MEC data
    analysis = detect_formation_type(
        formation_data['positions'],
        formation_data['velocities']
    )

    # Add metadata about data source
    if analysis.quality_metrics is None:
        analysis.quality_metrics = {}

    analysis.quality_metrics['data_source'] = 'MEC_ephemeris'
    analysis.quality_metrics['coordinate_system'] = 'GSM'
    analysis.quality_metrics['authoritative_source'] = True

    return analysis


def print_formation_analysis(analysis: FormationAnalysis, verbose: bool = True):
    """Print detailed formation analysis results"""

    print(f"\n🔍 MMS Formation Analysis Results")
    print("=" * 50)

    print(f"Formation Type: {analysis.formation_type.value.upper()}")
    print(f"Confidence: {analysis.confidence:.3f}")
    print(f"Formation Volume: {analysis.volume:.0f} km³")
    print(f"Characteristic Scale: {analysis.characteristic_scale:.1f} km")

    print(f"\nGeometric Properties:")
    print(f"  Linearity: {analysis.linearity:.3f}")
    print(f"  Planarity: {analysis.planarity:.3f}")
    print(f"  Sphericity: {analysis.sphericity:.3f}")

    print(f"\nPrincipal Components: [{analysis.principal_components[0]:.1f}, "
          f"{analysis.principal_components[1]:.1f}, {analysis.principal_components[2]:.1f}] km²")

    if verbose:
        print(f"\nSpacecraft Orderings:")
        for coord_sys, ordering in analysis.spacecraft_ordering.items():
            order_str = ' → '.join([f'MMS{p}' for p in ordering])
            print(f"  {coord_sys:8s}: {order_str}")

        print(f"\nInter-spacecraft Separations:")
        for pair, distance in analysis.separations.items():
            print(f"  MMS{pair}: {distance:.1f} km")

        print(f"\nQuality Metrics:")
        for metric, value in analysis.quality_metrics.items():
            print(f"  {metric}: {value:.3f}")

    # Recommendations
    recommended_method = get_formation_specific_analysis_method(analysis)
    print(f"\nRecommended Analysis Method: {recommended_method}")

    if analysis.formation_type == FormationType.STRING_OF_PEARLS:
        print("📝 String-of-Pearls Formation Detected:")
        print("   • Use 1D timing analysis along principal axis")
        print("   • Focus on boundary normal determination")
        print("   • Consider reduced spatial resolution")
    elif analysis.formation_type == FormationType.TETRAHEDRAL:
        print("📝 Tetrahedral Formation Detected:")
        print("   • Use full 3D gradient analysis")
        print("   • Apply standard MVA techniques")
        print("   • Excellent for boundary normal determination")
    elif analysis.formation_type == FormationType.PLANAR:
        print("📝 Planar Formation Detected:")
        print("   • Use 2D analysis in formation plane")
        print("   • Limited out-of-plane resolution")
        print("   • Good for in-plane gradients")
