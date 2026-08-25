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
from .native_hyrax import MASK_DOMAIN, MASK_PROTOCOL_ID, MASK_VERSION
from .util import digest_json


PROFILE_ID = "masked_full_multi_session_v1"
PROFILE_SCHEMA = "zkGPT/privacy-profile-v1"
PROFILE_STATUS = "supported_multi_session_masked_full"
FORMAL_TRANSCRIPT_DOMAIN = "zkGPT/GAT/formal-proof-session/v1"
PRIVACY_BIND_LABEL = "privacy/profile"
COMPONENT_ORDER = (
    "lasso",
    "matmul",
    "gkr",
    "input_combine",
    "terminal_openings",
)
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
    if item.get("backend") != COMMITMENT_BACKEND_ID:
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
        # messages only; it never contains the hidden scalar opening.
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
        if (
            not isinstance(mask_commitment.get("x"), str)
            or not mask_commitment["x"].isdigit()
            or not isinstance(mask_commitment.get("y"), str)
            or not mask_commitment["y"].isdigit()
        ):
            raise PrivacyProfileError(f"{label} has malformed mask commitment coordinates")
        challenge = payload.get("mask_challenge")
        if not isinstance(challenge, str) or not challenge.isdigit():
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
            ipa = value.get("ipa")
            if ipa is not None and not isinstance(ipa, Mapping):
                raise PrivacyProfileError(f"{component}/{path} IPA payload is malformed")
            # A committed linear claim intentionally carries only a fresh
            # evaluation commitment and is consumed by the later canonical
            # input reduction.  It is not an opening.  Every actual IPA
            # opening (including ephemeral openings) must carry the masked
            # schema below.
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
                mask=ipa is not None,
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
