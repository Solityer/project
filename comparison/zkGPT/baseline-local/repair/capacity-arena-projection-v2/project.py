#!/usr/bin/env python3
"""Analyze a challenge-time scalar compaction plan for zkGPT GKR arenas.

This is an analytical projection only. It does not modify or run the prover.
"""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "baseline-local/repair/capacity-analysis-symbolic-v4"
OUT_PATH = Path(__file__).with_name("projection.json")
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
FIELD_BYTES = 32
HEAP_LIMIT = 8 << 20


def round_up(value, unit):
    return ((value + unit - 1) // unit) * unit


def scalar_capacity(extent):
    return max(2, extent + (extent & 1))


def mapped_bytes(byte_count):
    return byte_count if byte_count > HEAP_LIMIT else 0


def punched_allocation_bound(active_bytes, file_bytes, block_size):
    if file_bytes <= HEAP_LIMIT or active_bytes == 0:
        return 0
    # Keep all live pages and one possible trailing partial allocation block.
    return min(file_bytes, round_up(active_bytes, block_size)
               + (block_size if file_bytes % block_size else 0))


def prefix_projection(profile, block_size):
    best_projected = {"bytes": 0}
    recomputed_current = 0
    largest_source_read_bytes = 0
    for layer_index, layer in enumerate(profile["gkr_layer_work_profile"]):
        if layer["layer_type"] == 11:  # FCONN uses its native MatMul path.
            continue
        for phase, rounds, lane_keys in (
            ("phase1", layer["max_bl_u"], ("size_u0", "size_u1")),
            ("phase2", layer["max_bl_v"], ("size_v0", "size_v1")),
        ):
            if rounds < 0:
                continue
            extents = [layer[key] for key in lane_keys]
            file_sizes = [scalar_capacity(n) * FIELD_BYTES for n in extents]
            # Current implementation keeps a value and multiplier arena for
            # both lanes, in packed affine form after the first challenge.
            current_phase = 2 * sum(mapped_bytes(size) for size in file_sizes)
            recomputed_current = max(recomputed_current, current_phase)

            # Proposed phase start allocates M only; the first V values are a
            # read-only view over the already-owned canonical witness.
            multiplier_only = sum(mapped_bytes(size) for size in file_sizes)
            phase_peak = multiplier_only
            phase_stage = "multiplier_only_before_first_challenge"
            challenge_compacted = 0
            phase_source_values = 0
            if rounds >= 1:
                for extent, file_size in zip(extents, file_sizes):
                    if extent == 0:
                        continue
                    folded_extent = (extent + 1) // 2
                    active_m_bytes = folded_extent * FIELD_BYTES
                    value_file_bytes = scalar_capacity(folded_extent) * FIELD_BYTES
                    m_after = punched_allocation_bound(
                        active_m_bytes, file_size, block_size)
                    v_after = mapped_bytes(value_file_bytes)
                    challenge_compacted += m_after + v_after
                    # Source is traversed once in round-0 polynomial assembly
                    # and once after the challenge to seed half-size V. The
                    # latter scan replaces delayed reads of the old full V
                    # prefix; it is real witness traffic and is not removed
                    # from the reported workload.
                    phase_source_values += extent
                if challenge_compacted > phase_peak:
                    phase_peak = challenge_compacted
                    phase_stage = "after_first_challenge_m_reclaim_then_half_v"
                largest_source_read_bytes += phase_source_values * FIELD_BYTES
            if phase_peak > best_projected["bytes"]:
                best_projected = {
                    "bytes": phase_peak,
                    "layer_index": layer_index,
                    "phase": phase,
                    "rounds": rounds,
                    "source_extents": extents,
                    "multiplier_only_bytes": multiplier_only,
                    "after_first_challenge_bytes": challenge_compacted,
                    "phase_peak_stage": phase_stage,
                }
    expected_current = profile["native_mapped_phase_breakdown"]["component:gkr_prefix"]
    if recomputed_current != expected_current:
        raise RuntimeError(
            f"current prefix mismatch: {recomputed_current} != {expected_current}")
    return best_projected, expected_current, largest_source_read_bytes


def read_mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def main():
    stat = os.statvfs(ROOT)
    block_size = max(os.sysconf("SC_PAGE_SIZE"), stat.f_frsize)
    available = stat.f_bavail * stat.f_frsize
    reserve = (available + 19) // 20
    report = {
        "projection_kind": "analysis_only_challenge_time_scalar_compaction",
        "implementation_validated": False,
        "protocol_or_relation_changed": False,
        "filesystem_block_bytes": block_size,
        "current_filesystem_available_bytes": available,
        "current_mem_available_bytes": read_mem_available(),
        "filesystem_reserve_5pct_available_bytes": reserve,
        "source_profiles": {},
        "datasets": {},
        "source_boundaries": [
            "All five inputs are latest-binary ANALYSIS_ONLY symbolic profiles; non-FCONN per-layer source extents are conservative bounds.",
            "Only the packed GKR value/multiplier arena lifetime is projected; gates, source maps, witness, transcript, other phases, and artifact bytes remain from the current source-bound report.",
            "After each transcript challenge, a scalar array of ceil(n/2) fields is written into the original M arena, its dead file tail is punched, and only then is the V scalar array allocated.",
            "The allocator mapping retains the original mmap/file length and registry size; only allocated disk blocks fall. Tail accounting keeps active pages and one possible short tail block.",
            "The round polynomial still reads immutable witness values for round zero; V is seeded from that source after the challenge. This adds a second witness scan of the active source extents and is explicitly counted.",
            "The proposed change is not implemented. No proving-time projection is made from counts.",
        ],
    }
    for dataset in DATASETS:
        path = PROFILE_DIR / f"{dataset}.json"
        profile = json.loads(path.read_text())
        if profile.get("profile_kind") != "analysis_only" or profile.get("status") != "ANALYSIS_ONLY":
            raise RuntimeError(f"{dataset} source profile is not analysis-only")
        projected_prefix, old_prefix, challenge_source_read_bytes = prefix_projection(profile, block_size)
        stages = profile["native_mapped_phase_breakdown"]
        projected_gkr = stages["gkr_proving"] - old_prefix + projected_prefix["bytes"]
        other_stages = {
            key: value for key, value in stages.items()
            if not key.startswith("component:") and key != "gkr_proving"
        }
        scratch_peak = max(projected_gkr, *other_stages.values())
        artifact = profile["artifact_disk_bytes_upper_bound"]
        requirement = scratch_peak + artifact
        raw_entries = profile["explicit_unary_gate_buffer_entries_upper_bound"]
        raw_bytes = profile["max_explicit_unary_gate_buffer_bytes_upper_bound"]
        report["source_profiles"][dataset] = {
            "profile_binary_sha256": profile["profile_binary_sha256"],
            "profile_status": profile["status"],
        }
        report["datasets"][dataset] = {
            "shape": {
                "nodes": profile["num_nodes"],
                "edges": profile["num_edges"],
                "features": profile["num_features"],
                "classes": profile["num_classes"],
                "source_domain_upper_bound": profile["gkr_layer0_source_domain_entries_upper_bound"],
                "source_domains_exact": profile["gkr_point_shape_exact"],
            },
            "gkr_work": {
                "phase1_rounds": profile["gkr_phase1_rounds_upper_bound"],
                "phase2_rounds": profile["gkr_phase2_rounds_upper_bound"],
                "phase1_fold_pairs": profile["gkr_phase1_total_pair_visits_upper_bound"],
                "phase2_fold_pairs": profile["gkr_phase2_total_pair_visits_upper_bound"],
                "fconn_scalar_multiply_terms": profile["fconn_scalar_multiply_terms"],
                "lasso_rows": profile["lasso_total_rows"],
                "lasso_row_chunks": profile["lasso_total_row_commitment_chunks"],
                "hyrax_matrix_openings": profile["hyrax_matrix_masked_openings"],
                "hyrax_lqry_openings": profile["hyrax_l_qry_tensor_openings"],
                "challenge_time_v_source_read_bytes_upper_bound": challenge_source_read_bytes,
            },
            "storage_bytes": {
                "current_gkr_prefix_mapped": old_prefix,
                "projected_gkr_prefix_mapped_peak": projected_prefix["bytes"],
                "largest_prefix_phase": projected_prefix,
                "current_gkr_proving_stage": stages["gkr_proving"],
                "projected_gkr_proving_stage": projected_gkr,
                "projected_other_stage_peaks": other_stages,
                "projected_scratch_peak": scratch_peak,
                "artifact_upper_bound": artifact,
                "scratch_plus_artifact_requirement": requirement,
                "filesystem_available_bytes": available,
                "filesystem_residual_after_requirement_bytes": available - requirement,
                "reserve_5pct_bytes": reserve,
                "meets_5pct_free_capacity_reserve": available - requirement >= reserve,
            },
            "host_memory_peak_bytes_from_profile": profile["estimated_peak_host_memory_bytes"],
        }
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
