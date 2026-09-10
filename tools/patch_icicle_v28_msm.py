#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one source match, found {count}"
        )

    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)

    text = source.read_text(
        encoding="utf-8"
    )

    old = """      // removing zero bucket, if it exists
      unsigned smallest_bucket_index;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &smallest_bucket_index, single_bucket_indices, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      // maybe zero bucket is empty after all? in this case zero_bucket_offset is set to 0
      unsigned zero_bucket_offset = (smallest_bucket_index == 0) ? 1 : 0;

      // sort by bucket sizes
      unsigned h_nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_buckets_to_compute, stream));
      h_nof_buckets_to_compute -= zero_bucket_offset;
"""

    new = """      // Both values are consumed immediately by host control flow.
      // Queue the D2H transfers on the producing stream and explicitly wait
      // for them before the CPU reads either stack variable.
      unsigned smallest_bucket_index;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &smallest_bucket_index, single_bucket_indices, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

      unsigned h_nof_buckets_to_compute;
      CHK_IF_RETURN(cudaMemcpyAsync(
        &h_nof_buckets_to_compute, nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

      CHK_IF_RETURN(cudaStreamSynchronize(stream));

      // maybe zero bucket is empty after all? in this case zero_bucket_offset is set to 0
      unsigned zero_bucket_offset = (smallest_bucket_index == 0) ? 1 : 0;
      CHK_IF_RETURN(cudaFreeAsync(nof_buckets_to_compute, stream));
      h_nof_buckets_to_compute -= zero_bucket_offset;
"""

    text = replace_once(
        text,
        old,
        new,
        "bucket host-control synchronization",
    )

    old = """      unsigned h_nof_large_buckets;
      CHK_IF_RETURN(
        cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_large_buckets, stream));
"""

    new = """      unsigned h_nof_large_buckets;
      CHK_IF_RETURN(
        cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaStreamSynchronize(stream));
      CHK_IF_RETURN(cudaFreeAsync(nof_large_buckets, stream));
"""

    text = replace_once(
        text,
        old,
        new,
        "large-bucket count synchronization",
    )

    old = """        unsigned h_nof_pts_in_large_buckets;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_nof_pts_in_large_buckets, sorted_bucket_sizes_sum + h_nof_large_buckets, sizeof(unsigned),
          cudaMemcpyDeviceToHost, stream_large_buckets));
        unsigned h_largest_bucket;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_largest_bucket, sorted_bucket_sizes, sizeof(unsigned), cudaMemcpyDeviceToHost, stream_large_buckets));
"""

    new = """        unsigned h_nof_pts_in_large_buckets;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_nof_pts_in_large_buckets, sorted_bucket_sizes_sum + h_nof_large_buckets, sizeof(unsigned),
          cudaMemcpyDeviceToHost, stream_large_buckets));
        unsigned h_largest_bucket;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &h_largest_bucket, sorted_bucket_sizes, sizeof(unsigned), cudaMemcpyDeviceToHost, stream_large_buckets));
        CHK_IF_RETURN(cudaStreamSynchronize(stream_large_buckets));
"""

    text = replace_once(
        text,
        old,
        new,
        "large-bucket metadata synchronization",
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        text,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()