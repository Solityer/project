#!/usr/bin/env python3
"""Project zkGPT prefix scratch after a two-fold borrowed-value design.

This is analysis only. It does not execute or modify the prover, and does not
claim the proposed value-view/tail-punch implementation is validated.
"""

import json
import os
from pathlib import Path


PROFILE_DIR = Path(__file__).resolve().parents[1] / "capacity-analysis-symbolic-v4"
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


def active_after_two_folds(extent):
    """Fields in ceil(ceil(extent/2)/2) affine entries."""
    return 2 * ((extent + 3) // 4)


def projected_prefix(profile, block_size):
    best = {"bytes": 0}
    current_max = 0
    borrowed_eval_count = 0
    borrowed_read_count = 0
    for layer_index, layer in enumerate(profile["gkr_layer_work_profile"]):
        if layer["layer_type"] == 11:  # FCONN has no GKR prefix arenas.
            continue
        for phase, rounds, lanes in (
            ("phase1", layer["max_bl_u"], ("size_u0", "size_u1")),
            ("phase2", layer["max_bl_v"], ("size_v0", "size_v1")),
        ):
            if rounds < 0:
                continue
            extents = [layer[key] for key in lanes]
            full_bytes = [scalar_capacity(n) * FIELD_BYTES for n in extents]
            current_phase = 2 * sum(mapped_bytes(size) for size in full_bytes)
            current_max = max(current_max, current_phase)

            # At phase start and for fewer than two folds, only M is allocated.
            multiplier_only = sum(mapped_bytes(size) for size in full_bytes)
            phase_peak = multiplier_only
            phase_stage = "multiplier_only"
            if rounds >= 2:
                post_fold_two = 0
                for extent, old_file_bytes in zip(extents, full_bytes):
                    if extent == 0:
                        continue
                    active_fields = active_after_two_folds(extent)
                    active_bytes = active_fields * FIELD_BYTES
                    if old_file_bytes > HEAP_LIMIT:
                        # The map/file length stays unchanged. Conservatively
                        # retain the active pages and one possible short tail
                        # block that hole punching cannot reclaim.
                        multiplier_reclaimed_bound = min(
                            old_file_bytes,
                            round_up(active_bytes, block_size)
                            + (block_size if old_file_bytes % block_size else 0),
                        )
                    else:
                        multiplier_reclaimed_bound = 0
                    value_output_mapping = mapped_bytes(active_bytes)
                    post_fold_two += multiplier_reclaimed_bound + value_output_mapping
                if post_fold_two > phase_peak:
                    phase_peak = post_fold_two
                    phase_stage = "after_second_fold_and_tail_reclaim"

                # The second fold reads all values in the first-fold active
                # extent lazily from the immutable original witness. Each
                # current value corresponds to two original values; a final
                # odd coordinate uses zero padding and one real source read.
                for extent in extents:
                    if extent:
                        folded_once = (extent + 1) // 2
                        borrowed_eval_count += folded_once
                        borrowed_read_count += extent

            if phase_peak > best["bytes"]:
                best = {
                    "bytes": phase_peak,
                    "layer_index": layer_index,
                    "phase": phase,
                    "rounds": rounds,
                    "source_extents": extents,
                    "multiplier_only_bytes": multiplier_only,
                    "phase_peak_stage": phase_stage,
                }
    if current_max != profile["native_mapped_phase_breakdown"]["component:gkr_prefix"]:
        raise RuntimeError(
            f"profile prefix mismatch: recomputed {current_max}, "
            f"profile reports {profile['native_mapped_phase_breakdown']['component:gkr_prefix']}"
        )
    return best, current_max, borrowed_eval_count, borrowed_read_count


def mem_available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def main():
    fs = os.statvfs(Path(__file__).resolve().parents[4])
    block_size = max(os.sysconf("SC_PAGE_SIZE"), fs.f_frsize)
    available = fs.f_bavail * fs.f_frsize
    results = {
        "projection_kind": "analysis_only_two_fold_borrowed_value_and_reclaim_model",
        "validated_implementation": False,
        "protocol_or_relation_changes": False,
        "assumptions": [
            "Source profiles are current-binary symbolic analysis-only profiles; non-FCONN source extents are conservative upper bounds.",
            "Value lanes read from the existing immutable NativeLayeredGkrWitness for the first two folds; no copied full value prefix exists then.",
            "A V affine arena is allocated only after fold two, when active fields are 2*ceil(n/4).",
            "M folds in place; after the existing worker/chunk barrier, dead mapped-file tails are hole-punched without changing logical mapping length or allocator registry bytes.",
            "Mapped M physical bound keeps active pages and one possible trailing partial block; fail-closed reclaim is required for admitted large path.",
            "Persistent witness, gates, source maps, input buffers, bounded scratch, transcript, proof artifact, and all non-prefix stage peaks remain unchanged.",
            "Projected scratch is the maximum existing capacity stage with only gkr_proving prefix bytes replaced by the two-fold projected prefix.",
            "The artifact upper bound is added to the peak scratch requirement; five percent of currently available filesystem bytes is held as reserve.",
        ],
        "page_or_filesystem_block_bytes": block_size,
        "current_filesystem_available_bytes": available,
        "current_mem_available_bytes": mem_available_bytes(),
        "source_profile_sha256": {},
        "datasets": {},
    }
    for dataset in DATASETS:
        path = PROFILE_DIR / f"{dataset}.json"
        profile = json.loads(path.read_text())
        if profile.get("profile_kind") != "analysis_only" or profile.get("status") != "ANALYSIS_ONLY":
            raise RuntimeError(f"{dataset} source profile is not analysis-only")
        projection, current_prefix, borrowed_evals, borrowed_reads = projected_prefix(profile, block_size)
        phases = profile["native_mapped_phase_breakdown"]
        projected_gkr = phases["gkr_proving"] - current_prefix + projection["bytes"]
        other_phase_peaks = {
            name: size for name, size in phases.items() if not name.startswith("component:") and name != "gkr_proving"
        }
        scratch_peak = max(projected_gkr, *other_phase_peaks.values())
        artifact = profile["artifact_disk_bytes_upper_bound"]
        total_requirement = scratch_peak + artifact
        reserve = (available + 19) // 20
        results["source_profile_sha256"][dataset] = profile["profile_binary_sha256"]
        results["datasets"][dataset] = {
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
                "borrowed_value_recomputations_at_second_fold": borrowed_evals,
                "borrowed_source_field_reads_at_second_fold": borrowed_reads,
            },
            "storage_bytes": {
                "current_prefix_mapped": current_prefix,
                "projected_prefix_mapped_peak": projection["bytes"],
                "projected_prefix_largest_phase": projection,
                "current_gkr_proving_stage": phases["gkr_proving"],
                "projected_gkr_proving_stage": projected_gkr,
                "projected_other_stage_peaks": other_phase_peaks,
                "projected_scratch_peak": scratch_peak,
                "artifact_upper_bound": artifact,
                "scratch_plus_artifact_requirement": total_requirement,
                "filesystem_available": available,
                "filesystem_reserve_5pct": reserve,
                "filesystem_residual_after_requirement": available - total_requirement,
                "residual_exceeds_5pct_reserve": available - total_requirement >= reserve,
            },
            "host_memory_peak_bytes_from_profile": profile["estimated_peak_host_memory_bytes"],
        }
    OUT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
