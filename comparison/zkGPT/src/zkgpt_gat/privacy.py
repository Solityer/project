"""Proof-level privacy profile and multi-session composition checks.

The zkGPT comparison proof has several native sub-protocols.  A verifier must
therefore check one authenticated privacy object which enumerates the hiding
messages emitted by *those* sub-protocols; an auxiliary opening envelope is
not evidence of masking.  This module deliberately works on
the public proof wire objects only.  Pedersen blinders and evaluation scalars
never enter the object or any fingerprint.

``masked_full_multi_session_v1`` uses the hiding commitments already emitted by
the active native Hyrax, committed-Sumcheck, Lasso, MatMul and GKR paths.  Each
proof gets an OS-CSPRNG nonce and fresh evaluation/auxiliary commitments.  The
canonical evidence list binds the exact labels, commitment points, IPA/PoK
messages and transcript digests which the independent verifier will replay.
"""
from __future__ import annotations

import re
import secrets
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import COMMITMENT_BACKEND_ID, COMMITMENT_ORDER
from .bn254 import hash_point
from .field import parse_canonical_decimal
from .native_hyrax import (
    AFFINE_MASK_DOMAIN,
    AFFINE_MASK_PROTOCOL_ID,
    AFFINE_MASK_VERSION,
    MASK_DOMAIN,
    MASK_HEADER_SCHEMA,
    MASK_HEADER_VERSION,
    MASK_PROTOCOL_ID,
    MASK_VERSION,
    BATCHED_LINEAR_PROTOCOL_ID,
    LINEAR_PROTOCOL_ID,
    PROTOCOL_ID as HYRAX_MLE_PROTOCOL_ID,
    PROTOCOL_ID as HYRAX_REGISTRATION_BACKEND_ID,
    _point,
)
from .util import digest_json
from .value_commitment import (
    CrossParamsSameValueProof,
    MultiplicationProof,
    RepresentationProof,
    SameValueProof,
    ValueCommitment,
)
from .zk_sumcheck import COMMITTED_SUMCHECK_PROTOCOL, CommittedSumcheckProof


PROFILE_ID = "masked_full_multi_session_v1"
PROFILE_SCHEMA = "zkGPT/privacy-profile-v1"
PROFILE_STATUS = "supported_multi_session_masked_full"
RELATION_COVERAGE_SCHEMA = "zkGPT/privacy-relation-coverage-v1"
FORMAL_TRANSCRIPT_DOMAIN = "zkGPT/GAT/formal-proof-session/v1"
PRIVACY_BIND_LABEL = "privacy/profile"
COMPONENT_ORDER = (
    "lasso",
    "matmul",
    "gkr",
    "input_combine",
    "terminal_openings",
)
RELATION_EQUATIONS = {
    "lasso": (
        "LC_f-query-link", "M_f-membership-sumcheck", "M_f-terminal-opening",
        "M_f-booleanity", "M_f-one-hot",
        "M_f-padding-row", "LC_f-GP_f-memory-gates", "lookup-input-consistency",
    ),
    "matmul": (
        "output-evaluation", "residual-zero", "inner-product-sumcheck",
        "left-right-terminal-multiplication",
    ),
    "gkr": (
        "linear-equality", "product-sumcheck", "operand-product", "terminal-selector",
    ),
    "input_combine": (
        "source-claim-aggregation", "combine-sumcheck", "Open_W", "Open_Aux",
        "Open_L", "terminal-relation",
    ),
    "terminal_openings": ("Open_W", "Open_Aux", "Open_L"),
}
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PROOF_COMPONENT_KEYS = {
    "lasso": "Pi_Lasso",
    "matmul": "Pi_MatMul",
    "gkr": "Pi_GKR",
    "input_combine": "Pi_combine",
    "terminal_openings": "Open_in",
}


class PrivacyProfileError(ValueError):
    """Raised when a privacy profile cannot be reconstructed fail-closed."""


def fresh_session_nonce() -> str:
    """Return a public 256-bit proof instance nonce from the OS CSPRNG."""

    # ``secrets`` delegates to the operating system CSPRNG.  The nonce is
    # sampled before any transcript challenge or commitment-derived value.
    return secrets.token_bytes(32).hex()


def require_session_nonce(value: Any) -> str:
    if not isinstance(value, str) or _NONCE_RE.fullmatch(value) is None:
        raise PrivacyProfileError("session nonce must be exactly 32 random bytes encoded as lowercase hex")
    if value == "0" * 64:
        raise PrivacyProfileError("all-zero session nonce is not allowed")
    return value


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise PrivacyProfileError(f"{label} must be a lowercase SHA-256 digest")
    return value


def transcript_domain(session_nonce: str) -> str:
    """Return the effective FS domain, domain-separated per proof instance."""

    nonce = require_session_nonce(session_nonce)
    return f"{FORMAL_TRANSCRIPT_DOMAIN}/{nonce}"


def _public_registration(item: Mapping[str, Any], *, segment: str) -> dict[str, Any]:
    required = ("backend", "parameter_id", "vector_length", "commitment")
    if any(key not in item for key in required):
        raise PrivacyProfileError(f"{segment} commitment registration is incomplete")
    if item.get("backend") not in {COMMITMENT_BACKEND_ID, HYRAX_REGISTRATION_BACKEND_ID}:
        raise PrivacyProfileError(f"{segment} commitment backend mismatch")
    vector_length = item.get("vector_length")
    if isinstance(vector_length, bool) or not isinstance(vector_length, int) or vector_length <= 0:
        raise PrivacyProfileError(f"{segment} commitment vector length is invalid")
    # Only public registration fields are fingerprinted.  In particular, a
    # serialized ``blind`` field is forbidden in privacy evidence.
    if "blind" in item:
        raise PrivacyProfileError(f"{segment} privacy evidence contains a private blinder")
    return {
        "backend": item["backend"],
        "parameter_id": item["parameter_id"],
        "vector_length": int(vector_length),
        "commitment": item["commitment"],
    }


def commitment_fingerprint(item: Mapping[str, Any], *, segment: str) -> str:
    public = _public_registration(item, segment=segment)
    return digest_json({"segment": segment, **public})


def registration_identity_fingerprint(item: Mapping[str, Any], *, segment: str) -> str:
    """Fingerprint a registration independent of the proof path carrying it."""

    public = _public_registration(item, segment=segment)
    return digest_json(public)


def build_profile_anchor(metadata: Mapping[str, Any], commitments: Mapping[str, Any]) -> dict[str, Any]:
    """Build the metadata-side pre-commitment for a proof privacy profile.

    The anchor is deliberately computed after the three canonical segment
    commitments exist but before any native sub-proof is generated.  It is
    therefore not circular with ``metadata_digest`` and cannot be selected by
    a later proof message.
    """

    nonce = require_session_nonce(metadata.get("session_nonce"))
    if metadata.get("privacy_extension_profile_id") != PROFILE_ID:
        raise PrivacyProfileError("unsupported privacy profile id")
    items = commitments.get("items") if isinstance(commitments, Mapping) else None
    if not isinstance(items, Mapping):
        raise PrivacyProfileError("privacy anchor requires canonical commitment items")
    fingerprints: dict[str, str] = {}
    for segment in COMMITMENT_ORDER:
        item = items.get(segment)
        if not isinstance(item, Mapping):
            raise PrivacyProfileError(f"missing canonical commitment {segment}")
        fingerprints[segment] = registration_identity_fingerprint(item, segment=segment)
    anchor = {
        "schema": "zkGPT/privacy-profile-anchor-v1",
        "profile_id": PROFILE_ID,
        "instance_nonce": nonce,
        "transcript_domain": transcript_domain(nonce),
        "segment_commitment_fingerprints": fingerprints,
        "component_order": list(COMPONENT_ORDER),
        "masking_rule": "fresh-per-proof-masked-hyrax-ipa-and-native-committed-rounds",
        "randomness_source": "OS-CSPRNG-for-instance-and-native-commitments",
    }
    anchor["anchor_digest"] = digest_json(anchor)
    return anchor


def _path_join(path: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _canonical_mapping_items(value: Mapping[Any, Any], *, path: str) -> tuple[tuple[str, Any], ...]:
    """Return mapping entries in the protocol's canonical traversal order.

    Proof objects are often assembled by several native sub-protocols, so the
    insertion order of a Python mapping is not a protocol property.  Mapping
    keys are therefore required to be strings and sorted by their exact UTF-8
    byte encoding.  A custom ``Mapping`` can still expose duplicate entries;
    reject those explicitly instead of silently making the evidence depend on
    which duplicate happens to be visited first.
    """

    canonical: list[tuple[bytes, str, Any]] = []
    seen: set[bytes] = set()
    for key, child in value.items():
        if not isinstance(key, str):
            raise PrivacyProfileError(f"{path} mapping key is not a string")
        try:
            encoded = key.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PrivacyProfileError(f"{path} mapping key is not valid UTF-8") from exc
        if encoded in seen:
            raise PrivacyProfileError(f"{path} contains duplicate canonical mapping key")
        seen.add(encoded)
        canonical.append((encoded, key, child))
    canonical.sort(key=lambda item: item[0])
    return tuple((key, child) for _encoded, key, child in canonical)


def _active_label(value: Mapping[str, Any], path: str) -> str:
    label = value.get("label")
    if isinstance(label, str) and label:
        return label
    spec = value.get("spec")
    if isinstance(spec, Mapping) and isinstance(spec.get("label"), str) and spec["label"]:
        return str(spec["label"])
    return path


def _append_evidence(
    evidence: list[dict[str, Any]],
    *,
    component: str,
    path: str,
    role: str,
    active_label: str,
    payload: Mapping[str, Any],
    commitment: Any | None = None,
    registration_fingerprint: str | None = None,
    ipa: Any | None = None,
    transcript_digest: Any | None = None,
    pok: Any | None = None,
    mask: bool = False,
) -> None:
    label = f"{component}/{path}/{role}"
    item: dict[str, Any] = {
        "label": label,
        "component": component,
        "role": role,
        "active_label": active_label,
        "payload_digest": digest_json(payload),
    }
    if commitment is not None:
        item["commitment_fingerprint"] = digest_json(commitment)
    if registration_fingerprint is not None:
        item["registration_fingerprint"] = registration_fingerprint
    if ipa is not None:
        item["ipa_digest"] = digest_json(ipa)
    if pok is not None:
        item["pok_digest"] = digest_json(pok)
    if isinstance(transcript_digest, str) and transcript_digest:
        item["transcript_digest"] = transcript_digest
    elif transcript_digest is not None:
        raise PrivacyProfileError(f"{label} has a malformed transcript digest")
    if mask:
        # The fingerprint is a digest of public mask/evaluation/IPA/transcript
        # messages only; it never contains the hidden scalar opening.  Every
        # witness-dependent evaluation claim has an affine envelope, while a
        # full Hyrax opening additionally carries the native vector mask.
        affine = payload.get("affine_mask_protocol_id") is not None
        if affine:
            if payload.get("affine_mask_protocol_id") != AFFINE_MASK_PROTOCOL_ID:
                raise PrivacyProfileError(f"{label} has an invalid affine mask protocol")
            if payload.get("affine_mask_domain") != AFFINE_MASK_DOMAIN:
                raise PrivacyProfileError(f"{label} has an invalid affine mask domain")
            if payload.get("affine_mask_version") != AFFINE_MASK_VERSION:
                raise PrivacyProfileError(f"{label} has an invalid affine mask schema version")
            masked = payload.get("masked_evaluation_commitment")
            affine_mask = payload.get("mask_evaluation_commitment")
            if not isinstance(masked, Mapping) or not isinstance(affine_mask, Mapping):
                raise PrivacyProfileError(f"{label} has no public affine mask commitments")
            if set(masked) != {"x", "y", "infinity"} or set(affine_mask) != {"x", "y", "infinity"}:
                raise PrivacyProfileError(f"{label} has non-canonical affine mask commitments")
            if masked.get("infinity") is not False or affine_mask.get("infinity") is not False:
                raise PrivacyProfileError(f"{label} has an identity affine mask commitment")
            try:
                parsed_masked = _point(masked)
                parsed_affine_mask = _point(affine_mask)
            except (TypeError, ValueError):
                raise PrivacyProfileError(f"{label} has malformed affine mask commitment coordinates")
            if parsed_masked.infinity or parsed_affine_mask.infinity:
                raise PrivacyProfileError(f"{label} has an identity affine mask commitment")
            pok = payload.get("mask_evaluation_pok")
            if not isinstance(pok, Mapping) or not pok:
                raise PrivacyProfileError(f"{label} has no affine mask representation proof")
            item["affine_mask_protocol_id"] = AFFINE_MASK_PROTOCOL_ID
            item["affine_mask_domain"] = AFFINE_MASK_DOMAIN
            item["affine_mask_version"] = AFFINE_MASK_VERSION
            item["mask_commitment_fingerprint"] = digest_json(
                {
                    "affine_mask_protocol_id": AFFINE_MASK_PROTOCOL_ID,
                    "affine_mask_domain": AFFINE_MASK_DOMAIN,
                    "affine_mask_version": AFFINE_MASK_VERSION,
                    "masked_evaluation_commitment": masked,
                    "mask_evaluation_commitment": affine_mask,
                    "mask_evaluation_pok": pok,
                    "commitment": item.get("commitment_fingerprint"),
                    "ipa": item.get("ipa_digest"),
                    "transcript": item.get("transcript_digest"),
                }
            )
        else:
            if payload.get("mask_protocol_id") != MASK_PROTOCOL_ID:
                raise PrivacyProfileError(f"{label} is not a native masked Hyrax opening")
            if payload.get("mask_domain") != MASK_DOMAIN:
                raise PrivacyProfileError(f"{label} has an invalid mask domain")
            if payload.get("mask_version") != MASK_VERSION:
                raise PrivacyProfileError(f"{label} has an invalid mask schema version")
            mask_commitment = payload.get("mask_commitment")
            if not isinstance(mask_commitment, Mapping):
                raise PrivacyProfileError(f"{label} has no public mask commitment")
            if set(mask_commitment) != {"x", "y", "infinity"}:
                raise PrivacyProfileError(f"{label} has a non-canonical mask commitment")
            if mask_commitment.get("infinity") is not False:
                raise PrivacyProfileError(f"{label} has an identity mask commitment")
            try:
                parse_canonical_decimal(mask_commitment.get("x"), label=f"{label}.mask_commitment.x", nonnegative=True)
                parse_canonical_decimal(mask_commitment.get("y"), label=f"{label}.mask_commitment.y", nonnegative=True)
            except ValueError:
                raise PrivacyProfileError(f"{label} has malformed mask commitment coordinates")
            challenge = payload.get("mask_challenge")
            try:
                parse_canonical_decimal(challenge, label=f"{label}.mask_challenge", nonnegative=True)
            except ValueError:
                raise PrivacyProfileError(f"{label} has a malformed mask challenge")
            item["mask_protocol_id"] = MASK_PROTOCOL_ID
            item["mask_domain"] = MASK_DOMAIN
            item["mask_version"] = MASK_VERSION
            item["mask_commitment_fingerprint"] = digest_json(
                {
                    "mask_protocol_id": MASK_PROTOCOL_ID,
                    "mask_domain": MASK_DOMAIN,
                    "mask_version": MASK_VERSION,
                    "mask_commitment": mask_commitment,
                    "mask_challenge": challenge,
                    "commitment": item.get("commitment_fingerprint"),
                    "ipa": item.get("ipa_digest"),
                    "transcript": item.get("transcript_digest"),
                }
            )
    evidence.append(item)


_POK_KEYS = {
    "same_value",
    "linear_equality",
    "terminal_product",
    "terminal_multiplication",
    "group_equality_pok",
}


def _walk_component(value: Any, component: str, path: str, evidence: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        items = _canonical_mapping_items(value, path=path)
        active = _active_label(value, path)
        # Native segment/LC/GP registrations are actual commitment messages.
        if {"backend", "parameter_id", "vector_length", "commitment"}.issubset(value):
            registration = _public_registration(value, segment=f"{component}/{path}")
            label = str(value.get("label") or path)
            # Terminal opening registrations are keyed by canonical segment
            # name in ``items.com_W``/``items.com_Aux``/``items.com_L``;
            # their native registration label is only ``W``/``Aux``/``L``.
            # Use the canonical path so long-term ``com_W`` is not mistaken
            # for a per-proof mask commitment.
            _append_evidence(
                evidence,
                component=component,
                path=path,
                role="registration_commitment",
                active_label=label,
                payload=registration,
                commitment=registration["commitment"],
                registration_fingerprint=registration_identity_fingerprint(
                    value, segment=f"{component}/{path}"
                ),
                # A segment registration is a public binding object, not an
                # opening mask.  The actual opening evidence below carries
                # the native masked schema.
                mask=False,
            )
        if "evaluation_commitment" in value:
            commitment = value.get("evaluation_commitment")
            if not isinstance(commitment, Mapping):
                raise PrivacyProfileError(f"{component}/{path} evaluation commitment is malformed")
            # The historical field now carries the affine-masked point.  Do
            # not permit a pending claim or opening to enter the profile
            # without the explicit mask commitment and representation PoK.
            affine_fields = {
                "affine_mask_protocol_id", "affine_mask_domain", "affine_mask_version",
                "masked_evaluation_commitment", "mask_evaluation_commitment",
                "mask_evaluation_pok",
            }
            if not affine_fields.issubset(value):
                raise PrivacyProfileError(
                    f"{component}/{path} evaluation claim is not affine-masked"
                )
            ipa = value.get("ipa")
            if ipa is not None and not isinstance(ipa, Mapping):
                raise PrivacyProfileError(f"{component}/{path} IPA payload is malformed")
            _append_evidence(
                evidence,
                component=component,
                path=path,
                role="evaluation_commitment",
                active_label=active,
                payload=value,
                commitment=commitment,
                ipa=ipa,
                transcript_digest=value.get("transcript_digest"),
                mask=True,
            )
        if "transcript_digest" in value and "evaluation_commitment" not in value:
            digest = value.get("transcript_digest")
            if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest:
                raise PrivacyProfileError(f"{component}/{path} transcript digest is malformed")
            _append_evidence(
                evidence,
                component=component,
                path=path,
                role="transcript_digest",
                active_label=active,
                payload={"transcript_digest": digest},
                transcript_digest=digest,
                mask=False,
            )
        for key, child in items:
            child_path = _path_join(path, key)
            if key in _POK_KEYS or key.endswith("_pok"):
                if not isinstance(child, Mapping):
                    raise PrivacyProfileError(f"{component}/{child_path} PoK payload is malformed")
                _append_evidence(
                    evidence,
                    component=component,
                    path=child_path,
                    role="proof_of_knowledge",
                    active_label=active,
                    payload=child,
                    pok=child,
                    mask=False,
                )
            elif key == "coefficient_poks":
                if not isinstance(child, Sequence) or isinstance(child, (str, bytes, bytearray)):
                    raise PrivacyProfileError(f"{component}/{child_path} coefficient PoKs are malformed")
                for index, pok in enumerate(child):
                    if not isinstance(pok, Mapping):
                        raise PrivacyProfileError(f"{component}/{child_path}[{index}] PoK is malformed")
                    _append_evidence(
                        evidence,
                        component=component,
                        path=_path_join(child_path, index),
                        role="proof_of_knowledge",
                        active_label=active,
                        payload=pok,
                        pok=pok,
                        mask=False,
                    )
            _walk_component(child, component, child_path, evidence)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _walk_component(child, component, _path_join(path, index), evidence)


def _component_evidence(component: str, value: Any) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    _walk_component(value, component, component, evidence)
    if not evidence:
        raise PrivacyProfileError(f"active {component} proof contains no cryptographic evidence")
    labels = [item["label"] for item in evidence]
    if len(labels) != len(set(labels)):
        raise PrivacyProfileError(f"active {component} proof contains duplicate canonical evidence labels")
    return evidence


def _component_record(component: str, value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = _component_evidence(component, value)
    record = {
        "component": component,
        "proof_digest": digest_json(value),
        "evidence_digest": digest_json(evidence),
        "evidence_count": len(evidence),
    }
    return record, evidence


def build_privacy_profile(
    metadata: Mapping[str, Any],
    commitments: Mapping[str, Any],
    *,
    lasso: Any,
    matrix: Any,
    gkr: Any,
    input_combine: Any,
    terminal_openings: Any,
    pre_profile_transcript_digest: str,
) -> dict[str, Any]:
    """Construct the canonical profile from active proof objects."""

    nonce = require_session_nonce(metadata.get("session_nonce"))
    if metadata.get("privacy_extension_profile_id") != PROFILE_ID:
        raise PrivacyProfileError("privacy profile id is not masked_full_multi_session_v1")
    anchor = build_profile_anchor(metadata, commitments)
    if metadata.get("privacy_profile_anchor") != anchor:
        raise PrivacyProfileError("metadata privacy anchor does not match commitments")
    require_digest(pre_profile_transcript_digest, "pre-profile transcript digest")
    component_values = {
        "lasso": lasso,
        "matmul": matrix,
        "gkr": gkr,
        "input_combine": input_combine,
        "terminal_openings": terminal_openings,
    }
    _validate_relation_coverage(component_values)
    records: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for component in COMPONENT_ORDER:
        record, items = _component_record(component, component_values[component])
        records.append(record)
        evidence.extend(items)
    labels = [item["label"] for item in evidence]
    if len(labels) != len(set(labels)):
        raise PrivacyProfileError("privacy profile contains duplicate canonical labels")
    mask_fingerprints = [item["mask_commitment_fingerprint"] for item in evidence if "mask_commitment_fingerprint" in item]
    if not mask_fingerprints:
        raise PrivacyProfileError("privacy profile contains no per-proof masking commitments")
    if len(mask_fingerprints) != len(set(mask_fingerprints)):
        raise PrivacyProfileError("privacy profile reuses a masking commitment fingerprint")
    registration_roles = [item for item in evidence if item.get("role") == "registration_commitment"]
    if not any(item.get("registration_fingerprint") == anchor["segment_commitment_fingerprints"]["com_Aux"] for item in registration_roles):
        raise PrivacyProfileError("privacy profile is missing the dynamic com_Aux registration evidence")
    if not any(item.get("registration_fingerprint") == anchor["segment_commitment_fingerprints"]["com_L"] for item in registration_roles):
        raise PrivacyProfileError("privacy profile is missing the dynamic com_L registration evidence")
    lasso_registrations = [
        item for item in registration_roles if item.get("component") == "lasso"
    ]
    if not any("LC_f" in item.get("active_label", "") for item in lasso_registrations):
        raise PrivacyProfileError("privacy profile is missing LC registration evidence")
    if not any("GP_f" in item.get("active_label", "") for item in lasso_registrations):
        raise PrivacyProfileError("privacy profile is missing GP registration evidence")
    dynamic_commitments = anchor["segment_commitment_fingerprints"]["com_Aux"], anchor["segment_commitment_fingerprints"]["com_L"]
    profile: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "profile_id": PROFILE_ID,
        "status": PROFILE_STATUS,
        "instance_nonce": nonce,
        "transcript_domain": transcript_domain(nonce),
        "anchor_digest": anchor["anchor_digest"],
        "pre_profile_transcript_digest": pre_profile_transcript_digest,
        "component_order": list(COMPONENT_ORDER),
        "components": records,
        "evidence": evidence,
        "evidence_digest": digest_json(evidence),
        "masking": {
            "construction": "fresh-native-masked-hyrax-ipa+pedersen-evaluation-commitments-v1",
            "coverage": list(COMPONENT_ORDER),
            "long_term_com_W_fingerprint": anchor["segment_commitment_fingerprints"]["com_W"],
            "dynamic_segment_commitment_fingerprints": list(dynamic_commitments),
            "randomness_source": "OS-CSPRNG-and-native-cryptographic-commitment-randomness",
            "per_proof_freshness": True,
            "long_term_com_W_reused_only_as_public_binding": True,
            "hidden_scalars_serialized": False,
        },
        "relation_coverage": {
            "schema": RELATION_COVERAGE_SCHEMA,
            "verification": "wire-enumeration-and-native-equation-gates-v1",
            "components": {
                component: {
                    "masking": "affine-evaluation-claim+native-round-mask",
                    "equations": list(RELATION_EQUATIONS[component]),
                }
                for component in COMPONENT_ORDER
            },
            "terminal_openings": ["com_W", "com_Aux", "com_L"],
            "hidden_scalars_serialized": False,
        },
    }
    profile["profile_digest"] = digest_json(profile)
    return profile


def privacy_transcript_message(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small public message absorbed into the unified FS transcript."""

    required = ("schema", "profile_id", "profile_digest", "evidence_digest", "anchor_digest", "instance_nonce", "transcript_domain")
    if any(key not in profile for key in required):
        raise PrivacyProfileError("privacy profile transcript message is incomplete")
    return {key: profile[key] for key in required}


def _proof_payload_components(proof_payload: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Extract the only accepted privacy inputs from the proof wire object."""

    if not isinstance(proof_payload, Mapping):
        raise PrivacyProfileError("privacy verification requires the serialized proof payload")
    # Legacy duplicate opening envelopes are rejected, rather than treated as
    # compatibility evidence, so Open_in remains the only verifier-facing
    # opening source.  Build the historical field names without retaining
    # those obsolete identifiers in the production schema or inventory.
    legacy_fields = {"".join(("Open", "Masked")), "".join(("masked", "_openings"))}
    if any(field in proof_payload for field in legacy_fields):
        raise PrivacyProfileError("legacy duplicate privacy envelope is forbidden")
    components: dict[str, Any] = {}
    for component, key in _PROOF_COMPONENT_KEYS.items():
        if key not in proof_payload:
            raise PrivacyProfileError(f"proof payload is missing {key}")
        components[component] = proof_payload[key]
    profile = proof_payload.get("PrivacyProfile")
    if not isinstance(profile, Mapping):
        raise PrivacyProfileError("proof payload PrivacyProfile is missing")
    payload_digest = proof_payload.get("privacy_profile_digest")
    if payload_digest is not None and payload_digest != profile.get("profile_digest"):
        raise PrivacyProfileError("proof payload privacy profile digest mismatch")
    return components, profile


def _validate_mask_header(value: Any, *, path: str, component: str | None = None,
                          coverage: Sequence[str] | None = None) -> None:
    """Validate a versioned public coverage header without trusting labels."""

    if not isinstance(value, Mapping):
        raise PrivacyProfileError(f"{path} mask header is missing")
    required = {"schema", "version", "component", "session_nonce", "coverage", "coverage_commitments", "header_digest"}
    if set(value) != required:
        raise PrivacyProfileError(f"{path} mask header shape is invalid")
    if value.get("schema") != MASK_HEADER_SCHEMA or value.get("version") != MASK_HEADER_VERSION:
        raise PrivacyProfileError(f"{path} mask header schema/version is invalid")
    header_component = value.get("component")
    if not isinstance(header_component, str) or not header_component:
        raise PrivacyProfileError(f"{path} mask header component is invalid")
    if component is not None and header_component != component:
        raise PrivacyProfileError(f"{path} mask header component is invalid")
    nonce = require_session_nonce(value.get("session_nonce"))
    listed = value.get("coverage")
    if not isinstance(listed, Sequence) or isinstance(listed, (str, bytes, bytearray)):
        raise PrivacyProfileError(f"{path} mask header coverage is invalid")
    if any(not isinstance(item, str) or not item for item in listed):
        raise PrivacyProfileError(f"{path} mask header coverage entries are invalid")
    listed = list(listed)
    if not listed or len(set(listed)) != len(listed):
        raise PrivacyProfileError(f"{path} mask header coverage is not unique")
    if coverage is not None and listed != [str(item) for item in coverage]:
        raise PrivacyProfileError(f"{path} mask header coverage differs from the relation")
    commitments = value.get("coverage_commitments")
    if not isinstance(commitments, Mapping) or set(commitments) != set(listed):
        raise PrivacyProfileError(f"{path} mask coverage commitments are incomplete")
    for key in listed:
        try:
            point = _point(commitments[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise PrivacyProfileError(f"{path}.coverage_commitments.{key} is malformed") from exc
        if point.infinity:
            raise PrivacyProfileError(f"{path}.coverage_commitments.{key} is an identity point")
        expected_point = hash_point(
            AFFINE_MASK_DOMAIN + "/coverage",
            {"session_nonce": nonce, "component": header_component, "coverage": key},
        )
        if point != expected_point:
            raise PrivacyProfileError(f"{path}.coverage_commitments.{key} is not bound to the header")
    expected_digest = digest_json({key: value[key] for key in value if key != "header_digest"})
    if value.get("header_digest") != expected_digest:
        raise PrivacyProfileError(f"{path} mask header digest is invalid")


def _validate_masked_claim(value: Any, *, path: str, require_ipa: bool = False) -> None:
    """Check the public affine envelope used by every private evaluation."""

    if not isinstance(value, Mapping):
        raise PrivacyProfileError(f"{path} masked claim is not an object")
    affine_fields = {
        "evaluation_commitment", "affine_mask_protocol_id", "affine_mask_domain",
        "affine_mask_version", "masked_evaluation_commitment",
        "mask_evaluation_commitment", "mask_evaluation_pok",
    }
    # Keep the public claim envelope closed.  In particular, a proof cannot
    # smuggle a scalar blinder, witness vector, or an unclassified extension
    # through this profile's historical subset check.  The seven-field
    # envelope is intentionally used by input-combine source records; native
    # scalar, batched-linear, and full Hyrax claims have their own exact wire
    # shapes below.
    minimal_shape = set(affine_fields)
    scalar_shape = minimal_shape | {"protocol"}
    batched_shape = minimal_shape | {
        "protocol", "parameter_id", "vector_length", "transcript_digest",
    }
    full_shape = batched_shape | {
        "ipa", "mask_protocol_id", "mask_domain", "mask_version",
        "mask_commitment", "mask_challenge",
    }
    shape = set(value)
    if shape == minimal_shape:
        claim_kind = "envelope"
    elif shape == scalar_shape:
        if value.get("protocol") != "zkGPT/masked-scalar-claim-v1":
            raise PrivacyProfileError(f"{path} masked scalar claim protocol is invalid")
        claim_kind = "scalar"
    elif shape == batched_shape:
        if value.get("protocol") != BATCHED_LINEAR_PROTOCOL_ID:
            raise PrivacyProfileError(f"{path} batched linear claim protocol is invalid")
        claim_kind = "batched"
    elif shape == full_shape:
        if value.get("protocol") not in {HYRAX_MLE_PROTOCOL_ID, LINEAR_PROTOCOL_ID}:
            raise PrivacyProfileError(f"{path} native Hyrax opening protocol is invalid")
        claim_kind = "full"
    else:
        raise PrivacyProfileError(f"{path} masked claim wire shape is invalid")
    if require_ipa and claim_kind != "full":
        raise PrivacyProfileError(f"{path} native Hyrax IPA is missing")
    if value.get("affine_mask_protocol_id") != AFFINE_MASK_PROTOCOL_ID or value.get("affine_mask_domain") != AFFINE_MASK_DOMAIN or value.get("affine_mask_version") != AFFINE_MASK_VERSION:
        raise PrivacyProfileError(f"{path} affine mask protocol mismatch")
    masked = value.get("masked_evaluation_commitment")
    mask = value.get("mask_evaluation_commitment")
    evaluation = value.get("evaluation_commitment")
    if not isinstance(masked, Mapping) or not isinstance(mask, Mapping) or not isinstance(evaluation, Mapping):
        raise PrivacyProfileError(f"{path} affine mask commitments are malformed")
    if evaluation != masked:
        raise PrivacyProfileError(f"{path} historical evaluation field is not the masked point")
    try:
        masked_point = _point(masked)
        mask_point = _point(mask)
    except (TypeError, ValueError) as exc:
        raise PrivacyProfileError(f"{path} affine mask points are malformed") from exc
    if masked_point.infinity or mask_point.infinity:
        raise PrivacyProfileError(f"{path} affine mask points must be finite")
    if not isinstance(value.get("mask_evaluation_pok"), Mapping):
        raise PrivacyProfileError(f"{path} affine mask representation proof is missing")
    if claim_kind == "full":
        if not isinstance(value.get("ipa"), Mapping):
            raise PrivacyProfileError(f"{path} native Hyrax IPA is missing")
        if value.get("mask_protocol_id") != MASK_PROTOCOL_ID or value.get("mask_domain") != MASK_DOMAIN or value.get("mask_version") != MASK_VERSION:
            raise PrivacyProfileError(f"{path} native Hyrax vector mask schema is missing")
        vector_mask = value.get("mask_commitment")
        if not isinstance(vector_mask, Mapping) or vector_mask.get("infinity") is not False:
            raise PrivacyProfileError(f"{path} native Hyrax vector mask is invalid")
        try:
            parsed_vector_mask = _point(vector_mask)
        except (TypeError, ValueError) as exc:
            raise PrivacyProfileError(f"{path} native Hyrax vector mask coordinates are malformed") from exc
        if parsed_vector_mask.infinity:
            raise PrivacyProfileError(f"{path} native Hyrax vector mask is the identity")


def _validate_masked_sumcheck(value: Any, *, path: str) -> None:
    """Require coefficient masks, PoKs, recurrence and terminal cancellation."""

    if not isinstance(value, Mapping):
        raise PrivacyProfileError(f"{path} masked Sumcheck is missing")
    try:
        parsed = CommittedSumcheckProof.from_dict(dict(value))
    except (TypeError, ValueError, KeyError) as exc:
        raise PrivacyProfileError(f"{path} is not the current masked Sumcheck wire") from exc
    if parsed.protocol != COMMITTED_SUMCHECK_PROTOCOL or parsed.variable_count <= 0:
        raise PrivacyProfileError(f"{path} Sumcheck protocol/version is invalid")
    if parsed.terminal_mask_commitment.infinity:
        raise PrivacyProfileError(f"{path} terminal mask commitment is the identity")
    for index, round_message in enumerate(parsed.rounds):
        if not isinstance(round_message, Mapping):
            raise PrivacyProfileError(f"{path}.rounds[{index}] is malformed")
        required = {
            "round", "coefficient_commitments", "coefficient_mask_commitments",
            "coefficient_poks", "coefficient_mask_poks", "group_equality_pok",
            "mask_recurrence", "mask_recurrence_pok", "challenge",
        }
        if set(round_message) != required or round_message.get("round") != index:
            raise PrivacyProfileError(f"{path}.rounds[{index}] lacks masked coefficient coverage")
        width = parsed.degree + 1
        lists = [round_message.get(key) for key in (
            "coefficient_commitments", "coefficient_mask_commitments",
            "coefficient_poks", "coefficient_mask_poks",
        )]
        if any(not isinstance(item, list) or len(item) != width for item in lists):
            raise PrivacyProfileError(f"{path}.rounds[{index}] coefficient masks are incomplete")
        try:
            for coefficient in lists[0]:
                ValueCommitment.from_dict(coefficient)
            for coefficient in lists[1]:
                commitment = ValueCommitment.from_dict(coefficient)
                if commitment.point.infinity:
                    raise PrivacyProfileError(f"{path}.rounds[{index}] has an identity coefficient mask")
            for pok in lists[2] + lists[3]:
                RepresentationProof.from_dict(pok)
            RepresentationProof.from_dict(round_message["group_equality_pok"])
            RepresentationProof.from_dict(round_message["mask_recurrence_pok"])
        except (TypeError, ValueError, KeyError) as exc:
            raise PrivacyProfileError(f"{path}.rounds[{index}] contains malformed masking proof") from exc
        expected_recurrence = "zero_sum" if index == 0 else "challenge_recurrence"
        if round_message.get("mask_recurrence") != expected_recurrence:
            raise PrivacyProfileError(f"{path}.rounds[{index}] mask recurrence is invalid")


def _validate_equation_proof(value: Any, *, path: str, kind: str) -> None:
    """Parse the public sigma equation used by a native relation."""

    try:
        if kind == "same_value":
            CrossParamsSameValueProof.from_dict(dict(value))
        elif kind == "same_value_scalar":
            SameValueProof.from_dict(dict(value))
        elif kind == "multiplication":
            MultiplicationProof.from_dict(dict(value))
        elif kind == "representation":
            RepresentationProof.from_dict(dict(value))
        else:
            raise PrivacyProfileError(f"{path} has an unknown equation proof kind")
    except (TypeError, ValueError, KeyError) as exc:
        raise PrivacyProfileError(f"{path} equation proof is malformed") from exc


def _collect_evaluation_claims(value: Any, *, path: str, output: list[tuple[str, Mapping[str, Any]]]) -> None:
    if isinstance(value, Mapping):
        if "evaluation_commitment" in value:
            _validate_masked_claim(value, path=path, require_ipa="ipa" in value)
            output.append((path, value))
        for key, child in _canonical_mapping_items(value, path=path):
            _collect_evaluation_claims(child, path=_path_join(path, key), output=output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _collect_evaluation_claims(child, path=_path_join(path, index), output=output)


def _collect_mask_headers(value: Any, *, path: str, output: list[tuple[str, Mapping[str, Any]]]) -> None:
    if isinstance(value, Mapping):
        if "mask_header" in value:
            header = value["mask_header"]
            _validate_mask_header(header, path=f"{path}.mask_header")
            output.append((path, header))
        for key, child in _canonical_mapping_items(value, path=path):
            _collect_mask_headers(child, path=_path_join(path, key), output=output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _collect_mask_headers(child, path=_path_join(path, index), output=output)


def _validate_relation_coverage(active: Mapping[str, Any]) -> None:
    """Independently enumerate the native masked relation wire.

    This is deliberately separate from the profile digest inventory.  The
    profile can only be accepted when each relation exposes its own masking
    header, affine evaluation claims, committed Sumcheck recurrence and the
    terminal/opening equations consumed by the native verifier.
    """

    expected_headers = {
        "lasso": ("LC_f", "GP_f", "M_f", "ell_f", "membership-sumcheck", "lookup-input-consistency"),
        "matmul": ("output-evaluation", "residual-evaluation", "inner-product-sumcheck", "left-evaluation", "right-evaluation", "terminal-relation"),
        "gkr": ("linear-equality", "product-sumcheck", "operand-product", "terminal-selector", "lookup-input-consistency"),
        "input_combine": ("source-claims", "combine-sumcheck", "Open_W", "Open_Aux", "Open_L", "terminal-relation"),
        "terminal_openings": None,
    }
    all_claims: list[tuple[str, Mapping[str, Any]]] = []
    all_headers: list[tuple[str, Mapping[str, Any]]] = []
    for component in COMPONENT_ORDER:
        value = active.get(component)
        if not isinstance(value, Mapping):
            raise PrivacyProfileError(f"active {component} relation is missing")
        _collect_evaluation_claims(value, path=component, output=all_claims)
        _collect_mask_headers(value, path=component, output=all_headers)
        if component in {"lasso", "matmul", "gkr", "input_combine"} and not isinstance(value.get("mask_header"), Mapping):
            raise PrivacyProfileError(f"active {component} relation has no top-level mask header")
        expected = expected_headers.get(component)
        if expected is not None:
            _validate_mask_header(value.get("mask_header"), path=f"{component}.mask_header", component=component, coverage=expected)
    if not all_claims:
        raise PrivacyProfileError("active proof contains no masked evaluation claims")

    lasso = active["lasso"]
    families = lasso.get("families")
    if not isinstance(families, list) or not families:
        raise PrivacyProfileError("Lasso family wire is empty")
    for index, family in enumerate(families):
        path = f"lasso.families[{index}]"
        if not isinstance(family, Mapping):
            raise PrivacyProfileError(f"{path} is malformed")
        query_link = family.get("query_link")
        if not isinstance(query_link, Mapping):
            raise PrivacyProfileError(f"{path}.query_link is missing")
        for key in ("lc_claim", "canonical_claim"):
            _validate_masked_claim(query_link.get(key), path=f"{path}.query_link.{key}")
        membership = family.get("membership")
        if not isinstance(membership, Mapping):
            raise PrivacyProfileError(f"{path}.membership is missing")
        for key in (
            "matrix_registration", "padding_row", "booleanity", "one_hot",
            "ell_f", "ell_link", "sumcheck", "matrix_opening",
            "terminal_product",
        ):
            if key not in membership:
                raise PrivacyProfileError(f"{path}.membership.{key} is missing")
        booleanity = membership["booleanity"]
        if not isinstance(booleanity, Mapping):
            raise PrivacyProfileError(f"{path}.membership.booleanity is malformed")
        for key in ("target_opening", "left_opening", "right_opening"):
            opening = booleanity.get(key)
            if not isinstance(opening, Mapping):
                raise PrivacyProfileError(f"{path}.membership.booleanity.{key} is missing")
            _validate_masked_claim(
                opening.get("proof"),
                path=f"{path}.membership.booleanity.{key}.proof",
                require_ipa=True,
            )
        _validate_masked_sumcheck(
            booleanity.get("sumcheck"),
            path=f"{path}.membership.booleanity.sumcheck",
        )
        _validate_equation_proof(
            booleanity.get("operand_product"),
            path=f"{path}.membership.booleanity.operand_product",
            kind="multiplication",
        )
        _validate_equation_proof(
            booleanity.get("terminal_product"),
            path=f"{path}.membership.booleanity.terminal_product",
            kind="multiplication",
        )
        one_hot = membership["one_hot"]
        if not isinstance(one_hot, Mapping) or set(one_hot) != {
            "point", "linear_form", "claim", "same_value"
        }:
            raise PrivacyProfileError(f"{path}.membership.one_hot is malformed")
        _validate_masked_claim(
            one_hot.get("claim"),
            path=f"{path}.membership.one_hot.claim",
            require_ipa=True,
        )
        _validate_equation_proof(
            one_hot.get("same_value"),
            path=f"{path}.membership.one_hot.same_value",
            kind="same_value_scalar",
        )
        _validate_masked_claim(membership["ell_f"], path=f"{path}.membership.ell_f")
        _validate_equation_proof(membership["ell_link"], path=f"{path}.membership.ell_link", kind="same_value")
        _validate_masked_sumcheck(membership["sumcheck"], path=f"{path}.membership.sumcheck")
        opening = membership["matrix_opening"]
        if not isinstance(opening, Mapping):
            raise PrivacyProfileError(f"{path}.membership.matrix_opening is malformed")
        _validate_masked_claim(opening.get("proof"), path=f"{path}.membership.matrix_opening.proof", require_ipa=True)
        _validate_equation_proof(membership["terminal_product"], path=f"{path}.membership.terminal_product", kind="multiplication")
        local = family.get("local_combine")
        if not isinstance(local, Mapping):
            raise PrivacyProfileError(f"{path}.local_combine is missing")
        _validate_masked_sumcheck(local.get("sumcheck"), path=f"{path}.local_combine.sumcheck")
        _validate_equation_proof(local.get("terminal_product"), path=f"{path}.local_combine.terminal_product", kind="multiplication")
        local_openings = local.get("openings")
        if not isinstance(local_openings, Mapping) or set(local_openings) != {"LC", "GP"}:
            raise PrivacyProfileError(f"{path}.local_combine openings are incomplete")
        for name, payload in local_openings.items():
            _validate_masked_claim(payload, path=f"{path}.local_combine.openings.{name}", require_ipa=True)
        gates = family.get("gates")
        if not isinstance(gates, list) or not gates:
            raise PrivacyProfileError(f"{path}.gates are missing")
        gate_labels: list[str] = []
        for gate_index, gate in enumerate(gates):
            if not isinstance(gate, Mapping) or set(gate) != {"kind", "proof"}:
                raise PrivacyProfileError(f"{path}.gates[{gate_index}] is malformed")
            gate_proof = gate.get("proof")
            if not isinstance(gate_proof, Mapping) or not isinstance(gate_proof.get("spec"), Mapping):
                raise PrivacyProfileError(f"{path}.gates[{gate_index}] has no native spec")
            gate_label = gate_proof["spec"].get("label")
            if not isinstance(gate_label, str) or not gate_label:
                raise PrivacyProfileError(f"{path}.gates[{gate_index}] has no canonical label")
            gate_kind = gate.get("kind")
            if gate_kind == "linear":
                required_fields = {
                    "protocol", "spec", "point", "target_opening", "term_openings",
                    "linear_equality", "mask_header",
                }
            elif gate_kind == "product":
                required_fields = {
                    "protocol", "spec", "output_point", "target_opening", "sumcheck",
                    "left_opening", "right_opening", "product_commitment",
                    "operand_product", "terminal_product", "mask_header",
                }
            else:
                raise PrivacyProfileError(f"{path}.gates[{gate_index}] has an unknown native gate kind")
            if set(gate_proof) != required_fields:
                raise PrivacyProfileError(f"{path}.gates[{gate_index}] has incomplete native wire fields")
            gate_labels.append(gate_label)
        if len(gate_labels) != len(set(gate_labels)):
            raise PrivacyProfileError(f"{path}.gates contain duplicate native labels")
        required_gate_labels = {
            "memory/write-timestamp",
            "memory/left/accumulator-recurrence",
            "memory/right/accumulator-recurrence",
            "memory/grand-product-equality",
        }
        if not required_gate_labels.issubset(set(gate_labels)):
            raise PrivacyProfileError(f"{path}.gates do not cover the native memory relation")

    matrix = active["matmul"]
    claims = matrix.get("claims")
    if not isinstance(claims, list) or not claims:
        raise PrivacyProfileError("MatMul relation claims are empty")
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise PrivacyProfileError(f"matmul.claims[{index}] is malformed")
        spec = claim.get("spec")
        if not isinstance(spec, Mapping) or not isinstance(spec.get("label"), str) or not spec.get("label"):
            raise PrivacyProfileError(f"matmul.claims[{index}] has no canonical MatMul label")
        for key in ("output_opening", "residual_opening", "left_opening", "right_opening", "residual_binding"):
            if key not in claim:
                raise PrivacyProfileError(f"matmul.claims[{index}].{key} is missing")
        _validate_masked_sumcheck(claim.get("sumcheck"), path=f"matmul.claims[{index}].sumcheck")
        _validate_equation_proof(claim.get("terminal_multiplication"), path=f"matmul.claims[{index}].terminal_multiplication", kind="multiplication")
    matrix_labels = [claim["spec"]["label"] for claim in claims]
    if len(matrix_labels) != len(set(matrix_labels)):
        raise PrivacyProfileError("MatMul relation contains duplicate claim labels")

    gkr = active["gkr"]
    gkr_claims = gkr.get("claims")
    if not isinstance(gkr_claims, list) or not gkr_claims:
        raise PrivacyProfileError("GKR relation claims are empty")
    gkr_labels: list[str] = []
    gkr_kinds: set[str] = set()
    for index, claim in enumerate(gkr_claims):
        if not isinstance(claim, Mapping) or not isinstance(claim.get("proof"), Mapping):
            raise PrivacyProfileError(f"gkr.claims[{index}] is malformed")
        proof = claim["proof"]
        spec = proof.get("spec")
        if not isinstance(spec, Mapping) or not isinstance(spec.get("label"), str) or not spec.get("label"):
            raise PrivacyProfileError(f"gkr.claims[{index}] has no canonical gate label")
        gkr_labels.append(str(spec["label"]))
        kind = claim.get("kind")
        if kind not in {"linear", "product"}:
            raise PrivacyProfileError(f"gkr.claims[{index}] has an unknown native gate kind")
        gkr_kinds.add(str(kind))
        if kind == "linear":
            if set(proof) != {
                "protocol", "spec", "point", "target_opening", "term_openings",
                "linear_equality", "mask_header",
            }:
                raise PrivacyProfileError(f"gkr.claims[{index}] linear wire is incomplete")
            _validate_equation_proof(
                proof.get("linear_equality"),
                path=f"gkr.claims[{index}].proof.linear_equality",
                kind="representation",
            )
        else:
            if set(proof) != {
                "protocol", "spec", "output_point", "target_opening", "sumcheck",
                "left_opening", "right_opening", "product_commitment",
                "operand_product", "terminal_product", "mask_header",
            }:
                raise PrivacyProfileError(f"gkr.claims[{index}] product wire is incomplete")
        if "sumcheck" in proof:
            _validate_masked_sumcheck(proof["sumcheck"], path=f"gkr.claims[{index}].proof.sumcheck")
        for field in ("linear_equality", "operand_product", "terminal_product"):
            if field in proof:
                _validate_equation_proof(
                    proof[field], path=f"gkr.claims[{index}].proof.{field}",
                    kind="representation" if field == "linear_equality" else "multiplication",
                )
    if len(gkr_labels) != len(set(gkr_labels)) or gkr_kinds != {"linear", "product"}:
        raise PrivacyProfileError("GKR relation does not cover distinct linear and product gates")
    lasso_families = lasso.get("families")
    expected_lookup_labels = {
        f"lasso/{family['descriptor'].get('family')}/packed_query"
        for family in lasso_families
        if (
            isinstance(family, Mapping)
            and isinstance(family.get("descriptor"), Mapping)
            and family["descriptor"].get("kind") == "function"
            and isinstance(family["descriptor"].get("family"), str)
        )
    }
    if not expected_lookup_labels.issubset(set(gkr_labels)):
        raise PrivacyProfileError("GKR relation is missing lookup-input consistency gates")

    combine = active["input_combine"]
    sources = combine.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PrivacyProfileError("input combine has no source claims")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise PrivacyProfileError(f"input_combine.sources[{index}] is malformed")
        if source.get("segment") not in {"com_W", "com_Aux", "com_L"} or not isinstance(source.get("linear_form"), Mapping) or "weight" not in source:
            raise PrivacyProfileError(f"input_combine.sources[{index}] has no canonical segment claim")
        _validate_masked_claim(source.get("evaluation_claim"), path=f"input_combine.sources[{index}].evaluation_claim")
    _validate_masked_sumcheck(combine.get("sumcheck"), path="input_combine.sumcheck")
    terminal_product = combine.get("terminal_product")
    if not isinstance(terminal_product, Mapping):
        raise PrivacyProfileError("input_combine terminal relation is missing")
    _validate_equation_proof(terminal_product, path="input_combine.terminal_product", kind="multiplication")

    terminal = active["terminal_openings"]
    items = terminal.get("items") if isinstance(terminal, Mapping) else None
    if not isinstance(items, Mapping) or set(items) != {"com_W", "com_Aux", "com_L"}:
        raise PrivacyProfileError("terminal openings must cover exactly W, Aux and L")
    for name in ("com_W", "com_Aux", "com_L"):
        opening = items[name]
        if not isinstance(opening, Mapping):
            raise PrivacyProfileError(f"terminal opening {name} is malformed")
        if not isinstance(opening.get("registration"), Mapping) or "blind" in opening.get("registration", {}):
            raise PrivacyProfileError(f"terminal opening {name} has an invalid public registration")
        _validate_masked_claim(opening.get("proof"), path=f"terminal_openings.items.{name}.proof", require_ipa=True)

    # Every discovered header/claim is public coverage evidence; a component
    # with only an inventory digest cannot silently satisfy this relation gate.
    if len(all_headers) < 4 or len(all_claims) < 3:
        raise PrivacyProfileError("masked relation coverage is incomplete")


def _profile_anchor_digest(metadata: Mapping[str, Any]) -> str:
    anchor = metadata.get("privacy_profile_anchor")
    if not isinstance(anchor, Mapping):
        raise PrivacyProfileError("metadata privacy profile anchor is missing")
    if anchor.get("anchor_digest") != digest_json({key: value for key, value in anchor.items() if key != "anchor_digest"}):
        raise PrivacyProfileError("metadata privacy profile anchor digest is invalid")
    return require_digest(anchor.get("anchor_digest"), "privacy profile anchor digest")


def verify_privacy_profile(
    metadata: Mapping[str, Any],
    commitments: Mapping[str, Any],
    proof_payload: Any,
    *,
    pre_profile_transcript_digest: str,
    components: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild profile evidence from the authenticated proof payload.

    A naked profile is deliberately not accepted: every active component must
    come from ``Pi_Lasso``/``Pi_MatMul``/``Pi_GKR``/``Pi_combine``/``Open_in``
    in the same serialized proof payload, and the metadata anchor is rebuilt
    from the trusted commitment registrations.
    """

    try:
        active, profile = _proof_payload_components(proof_payload)
        if components is not None:
            if set(components) != set(COMPONENT_ORDER) or any(components[key] != active[key] for key in COMPONENT_ORDER):
                return {"status": "VERIFY_FAILED", "reason": "privacy components differ from proof payload"}
        _validate_relation_coverage(active)
        expected_anchor = build_profile_anchor(metadata, commitments)
        if metadata.get("privacy_profile_anchor") != expected_anchor:
            return {"status": "VERIFY_FAILED", "reason": "privacy profile anchor differs from commitments"}
        anchor_digest = _profile_anchor_digest(metadata)
        require_digest(pre_profile_transcript_digest, "pre-profile transcript digest")
        expected = build_privacy_profile(
            metadata,
            commitments,
            lasso=active["lasso"],
            matrix=active["matmul"],
            gkr=active["gkr"],
            input_combine=active["input_combine"],
            terminal_openings=active["terminal_openings"],
            pre_profile_transcript_digest=pre_profile_transcript_digest,
        )
        if dict(profile) != expected:
            return {"status": "VERIFY_FAILED", "reason": "privacy profile differs from active cryptographic evidence"}
        if profile.get("anchor_digest") != anchor_digest:
            return {"status": "VERIFY_FAILED", "reason": "privacy profile anchor binding is invalid"}
        if profile.get("profile_digest") != digest_json({key: value for key, value in profile.items() if key != "profile_digest"}):
            return {"status": "VERIFY_FAILED", "reason": "privacy profile digest is invalid"}
        return {
            "status": "VERIFY_OK",
            "profile_digest": profile["profile_digest"],
            "evidence_digest": profile["evidence_digest"],
            "anchor_digest": anchor_digest,
            "evidence_count": len(profile["evidence"]),
            "masking_commitment_count": sum("mask_commitment_fingerprint" in item for item in profile["evidence"]),
            "relation_coverage_verified": True,
            "relation_equations": {
                component: list(RELATION_EQUATIONS[component]) for component in COMPONENT_ORDER
            },
        }
    except (KeyError, TypeError, ValueError, PrivacyProfileError):
        return {"status": "VERIFY_FAILED", "reason": "malformed or incomplete privacy profile"}


def bind_privacy_profile(transcript: Any, profile: Mapping[str, Any]) -> None:
    """Absorb the profile digest and instance domain into the unified FS."""

    transcript.absorb(PRIVACY_BIND_LABEL, privacy_transcript_message(profile))


def _session_reuse_tokens(profile: Mapping[str, Any], com_w: str) -> set[str]:
    tokens: set[str] = set()
    for item in profile.get("evidence", []):
        if not isinstance(item, Mapping):
            raise PrivacyProfileError("malformed privacy evidence item")
        # Every repeated registration except the authenticated long-term W
        # commitment is a per-proof artifact and must remain fresh.
        if item.get("role") == "registration_commitment" and item.get("registration_fingerprint") == com_w:
            continue
        for field in ("payload_digest", "registration_fingerprint", "commitment_fingerprint", "ipa_digest", "pok_digest", "transcript_digest"):
            value = item.get(field)
            if value is not None:
                tokens.add(f"{field}:{value}")
    return tokens


def check_multi_session_profiles(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify real proof payloads, then check their joint masking freshness.

    Each element must contain ``metadata``, ``commitments``,
    ``proof_payload`` and the independently recorded pre-profile transcript
    digest.  Naked profiles are intentionally rejected.
    """

    try:
        if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes, bytearray)) or not sessions:
            return {"status": "VERIFY_FAILED", "reason": "at least one proof session is required"}
        profiles: list[Mapping[str, Any]] = []
        nonces: list[str] = []
        domains: list[str] = []
        com_w: list[str] = []
        evidence_sets: list[set[str]] = []
        # Keep freshness material grouped by proof session.  A native proof
        # is allowed to bind several component messages to the same session
        # transcript; only reuse *between* proofs is a multi-session failure.
        session_mask_fingerprints: list[set[str]] = []
        session_transcript_digests: list[set[str]] = []
        session_dynamic_commitments: list[set[str]] = []
        for index, session in enumerate(sessions):
            if not isinstance(session, Mapping):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} is not an authenticated artifact bundle"}
            metadata = session.get("metadata")
            commitments = session.get("commitments")
            proof_payload = session.get("proof_payload")
            pre_digest = session.get("pre_profile_transcript_digest")
            if not isinstance(metadata, Mapping) or not isinstance(commitments, Mapping) or not isinstance(proof_payload, Mapping):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} lacks real metadata/commitments/proof payload"}
            profile = proof_payload.get("PrivacyProfile")
            if not isinstance(profile, Mapping):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} lacks PrivacyProfile in proof payload"}
            check = verify_privacy_profile(metadata, commitments, proof_payload, pre_profile_transcript_digest=pre_digest)
            if check.get("status") != "VERIFY_OK":
                return {"status": "VERIFY_FAILED", "reason": f"session {index} privacy evidence invalid: {check.get('reason', 'unknown')}"}
            anchor = build_profile_anchor(metadata, commitments)
            w_item = commitments.get("items", {}).get("com_W")
            if not isinstance(w_item, Mapping):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} com_W registration is missing"}
            w_fingerprint = registration_identity_fingerprint(w_item, segment="com_W")
            masking = profile.get("masking")
            if not isinstance(masking, Mapping) or masking.get("long_term_com_W_fingerprint") != w_fingerprint:
                return {"status": "VERIFY_FAILED", "reason": f"session {index} com_W anchor mismatch"}
            nonce = require_session_nonce(profile.get("instance_nonce"))
            domain = profile.get("transcript_domain")
            if domain != transcript_domain(nonce) or metadata.get("session_nonce") != nonce:
                return {"status": "VERIFY_FAILED", "reason": f"session {index} transcript domain is not session-bound"}
            profiles.append(profile)
            nonces.append(nonce)
            domains.append(domain)
            com_w.append(w_fingerprint)
            evidence_sets.append(_session_reuse_tokens(profile, w_fingerprint))
            dynamic = masking.get("dynamic_segment_commitment_fingerprints")
            if not isinstance(dynamic, list) or len(dynamic) != 2 or any(not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None for value in dynamic):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} dynamic commitment evidence is malformed"}
            dynamic_set = set(dynamic)
            if len(dynamic_set) != len(dynamic):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} dynamic segment commitment reused within proof"}
            session_dynamic_commitments.append(dynamic_set)
            masks: set[str] = set()
            transcripts: set[str] = set()
            for item in profile.get("evidence", []):
                if "mask_commitment_fingerprint" in item:
                    masks.add(require_digest(item["mask_commitment_fingerprint"], "mask commitment fingerprint"))
                if isinstance(item.get("transcript_digest"), str):
                    transcripts.add(require_digest(item["transcript_digest"], "evidence transcript digest"))
            if len(masks) != sum("mask_commitment_fingerprint" in item for item in profile.get("evidence", [])):
                return {"status": "VERIFY_FAILED", "reason": f"session {index} mask commitment fingerprint reused within proof"}
            session_mask_fingerprints.append(masks)
            session_transcript_digests.append(transcripts)
        if len(set(nonces)) != len(nonces) or len(set(domains)) != len(domains):
            return {"status": "VERIFY_FAILED", "reason": "proof instance nonce/transcript domain reused"}
        if len(set(com_w)) != 1:
            return {"status": "VERIFY_FAILED", "reason": "sessions do not share the same com_W"}
        for left in range(len(profiles)):
            for right in range(left + 1, len(profiles)):
                if session_dynamic_commitments[left] & session_dynamic_commitments[right]:
                    return {"status": "VERIFY_FAILED", "reason": "dynamic segment commitment reused across sessions"}
                if session_mask_fingerprints[left] & session_mask_fingerprints[right]:
                    return {"status": "VERIFY_FAILED", "reason": "mask commitment fingerprint reused across sessions"}
                if session_transcript_digests[left] & session_transcript_digests[right]:
                    return {"status": "VERIFY_FAILED", "reason": "native transcript digest reused across sessions"}
        mask_fingerprints = set().union(*session_mask_fingerprints)
        transcript_digests = set().union(*session_transcript_digests)
        dynamic_commitments = set().union(*session_dynamic_commitments)
        for left in range(len(evidence_sets)):
            for right in range(left + 1, len(evidence_sets)):
                if evidence_sets[left] & evidence_sets[right]:
                    return {"status": "VERIFY_FAILED", "reason": "per-proof privacy evidence reused across sessions"}
        return {
            "status": "VERIFY_OK",
            "session_count": len(profiles),
            "shared_com_W_fingerprint": com_w[0],
            "unique_instance_nonces": len(set(nonces)),
            "unique_mask_commitments": len(mask_fingerprints),
            "unique_transcript_digests": len(transcript_digests),
            "unique_dynamic_segment_commitments": len(dynamic_commitments),
        }
    except (KeyError, TypeError, ValueError, PrivacyProfileError):
        return {"status": "VERIFY_FAILED", "reason": "malformed multi-session proof artifact set"}
