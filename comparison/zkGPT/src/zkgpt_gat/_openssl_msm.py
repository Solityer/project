"""Optional native BN254 MSM accelerator for the zkGPT CPU path.

The baseline keeps the Python :class:`~zkgpt_gat.bn254.G1Point` wire type and
all protocol equations unchanged.  When the host provides OpenSSL's generic
prime-field EC implementation, this small ctypes adapter evaluates one
multi-scalar multiplication in native code and converts the result back to
the same affine point type.  It is deliberately an optional arithmetic
accelerator: setup, transcript, serialization, and verifier code do not
depend on OpenSSL-specific encodings or APIs.

OpenSSL 3 exposes the BN254 short-Weierstrass curve through the generic
``EC_GROUP_new_curve_GFp``/``EC_POINTs_mul`` API even though BN254 is not one
of its named curves.  The caller validates the input points with the Python
curve checks before entering this module; this module also asks OpenSSL to
validate each affine coordinate before using it.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
from typing import Any, Sequence


_Void = ctypes.c_void_p
_Int = ctypes.c_int
_Size = ctypes.c_size_t

_FQ_MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583
_FR_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class OpenSSLMSMUnavailable(RuntimeError):
    """Raised when the optional native MSM cannot be initialized."""


class _OpenSSLMSM:
    """Own one immutable OpenSSL BN254 group and serialize calls safely."""

    _POINT_CACHE_LIMIT = 16_384

    def __init__(self) -> None:
        if os.environ.get("ZKGPT_DISABLE_OPENSSL_MSM", "").strip() == "1":
            raise OpenSSLMSMUnavailable("OpenSSL MSM disabled by environment")
        library_name = ctypes.util.find_library("crypto")
        if not library_name:
            raise OpenSSLMSMUnavailable("libcrypto is not available")
        try:
            lib = ctypes.CDLL(library_name)
            self._configure(lib)
            context = lib.BN_CTX_new()
            if not context:
                raise OpenSSLMSMUnavailable("BN_CTX_new failed")
            self._lib = lib
            self._context = context
            p = self._bn(_FQ_MODULUS)
            a = self._bn(0)
            b = self._bn(3)
            try:
                group = lib.EC_GROUP_new_curve_GFp(p, a, b, context)
            finally:
                lib.BN_free(p)
                lib.BN_free(a)
                lib.BN_free(b)
            if not group:
                raise OpenSSLMSMUnavailable("EC_GROUP_new_curve_GFp failed")
            self._group = group
            self._lock = threading.RLock()
            # Hyrax repeatedly combines the same generator sets.  Keeping
            # their native affine points avoids re-allocating and validating
            # every point for every IPA round.  The cache is bounded so a
            # caller cannot turn this optional accelerator into unbounded
            # native-memory growth; vectors beyond the bound use ephemeral
            # points for that call only.
            self._point_cache: dict[tuple[int, int], _Void] = {}
        except OpenSSLMSMUnavailable:
            raise
        except (AttributeError, OSError, TypeError) as exc:
            raise OpenSSLMSMUnavailable("OpenSSL EC MSM symbols are unavailable") from exc

    @staticmethod
    def _configure(lib: Any) -> None:
        """Declare only the stable OpenSSL C symbols used by this adapter."""

        lib.BN_CTX_new.restype = _Void
        lib.BN_CTX_new.argtypes = []
        lib.BN_CTX_free.restype = None
        lib.BN_CTX_free.argtypes = [_Void]
        lib.BN_new.restype = _Void
        lib.BN_new.argtypes = []
        lib.BN_bin2bn.restype = _Void
        lib.BN_bin2bn.argtypes = [ctypes.c_void_p, _Int, _Void]
        lib.BN_bn2binpad.restype = _Int
        lib.BN_bn2binpad.argtypes = [_Void, ctypes.c_void_p, _Int]
        lib.BN_free.restype = None
        lib.BN_free.argtypes = [_Void]

        lib.EC_GROUP_new_curve_GFp.restype = _Void
        lib.EC_GROUP_new_curve_GFp.argtypes = [_Void, _Void, _Void, _Void]
        lib.EC_GROUP_free.restype = None
        lib.EC_GROUP_free.argtypes = [_Void]
        lib.EC_POINT_new.restype = _Void
        lib.EC_POINT_new.argtypes = [_Void]
        lib.EC_POINT_free.restype = None
        lib.EC_POINT_free.argtypes = [_Void]
        lib.EC_POINT_set_affine_coordinates.restype = _Int
        lib.EC_POINT_set_affine_coordinates.argtypes = [_Void, _Void, _Void, _Void, _Void]
        lib.EC_POINT_get_affine_coordinates.restype = _Int
        lib.EC_POINT_get_affine_coordinates.argtypes = [_Void, _Void, _Void, _Void, _Void]
        lib.EC_POINTs_mul.restype = _Int
        lib.EC_POINTs_mul.argtypes = [
            _Void,
            _Void,
            _Void,
            _Size,
            ctypes.POINTER(_Void),
            ctypes.POINTER(_Void),
            _Void,
        ]

    def _bn(self, value: int) -> _Void:
        encoded = ctypes.create_string_buffer(int(value).to_bytes(32, "big"))
        result = self._lib.BN_bin2bn(encoded, 32, None)
        if not result:
            raise OpenSSLMSMUnavailable("BN_bin2bn failed")
        return result

    def msm(self, entries: Sequence[tuple[Any, int]]) -> Any:
        """Return the native MSM result as a Python ``G1Point``."""

        if not entries:
            # Import lazily to avoid a module import cycle during bn254 import.
            from .bn254 import G1Point

            return G1Point.zero()
        if any(int(scalar) < 0 or int(scalar) >= _FR_MODULUS for _point, scalar in entries):
            raise ValueError("BN254 MSM scalar is outside canonical Fr")
        points: list[_Void] = []
        ephemeral_points: list[_Void] = []
        local_points: dict[tuple[int, int], _Void] = {}
        scalars: list[_Void] = []
        output: _Void | None = None
        output_x: _Void | None = None
        output_y: _Void | None = None
        try:
            with self._lock:
                for point, scalar in entries:
                    key = (int(point.x), int(point.y))
                    native_point = self._point_cache.get(key)
                    if native_point is None:
                        native_point = local_points.get(key)
                    if native_point is None:
                        x = self._bn(key[0])
                        y = self._bn(key[1])
                        native_point = self._lib.EC_POINT_new(self._group)
                        if not native_point:
                            self._lib.BN_free(x)
                            self._lib.BN_free(y)
                            raise OpenSSLMSMUnavailable("EC_POINT_new failed")
                        try:
                            if self._lib.EC_POINT_set_affine_coordinates(
                                self._group, native_point, x, y, self._context
                            ) != 1:
                                raise ValueError("OpenSSL rejected a BN254 affine point")
                        except Exception:
                            self._lib.EC_POINT_free(native_point)
                            raise
                        finally:
                            self._lib.BN_free(x)
                            self._lib.BN_free(y)
                        local_points[key] = native_point
                        if len(self._point_cache) < self._POINT_CACHE_LIMIT:
                            self._point_cache[key] = native_point
                        else:
                            ephemeral_points.append(native_point)
                    points.append(native_point)
                    scalars.append(self._bn(int(scalar)))

                point_array = (_Void * len(points))(*points)
                scalar_array = (_Void * len(scalars))(*scalars)
                output = self._lib.EC_POINT_new(self._group)
                if not output:
                    raise OpenSSLMSMUnavailable("EC_POINT_new output failed")
                if self._lib.EC_POINTs_mul(
                    self._group,
                    output,
                    None,
                    len(points),
                    point_array,
                    scalar_array,
                    self._context,
                ) != 1:
                    raise OpenSSLMSMUnavailable("EC_POINTs_mul failed")
                output_x = self._lib.BN_new()
                output_y = self._lib.BN_new()
                if not output_x or not output_y:
                    raise OpenSSLMSMUnavailable("BN_new output failed")
                if self._lib.EC_POINT_get_affine_coordinates(
                    self._group, output, output_x, output_y, self._context
                ) != 1:
                    raise OpenSSLMSMUnavailable("EC_POINT_get_affine_coordinates failed")
                x_buffer = ctypes.create_string_buffer(32)
                y_buffer = ctypes.create_string_buffer(32)
                if (
                    self._lib.BN_bn2binpad(output_x, x_buffer, 32) != 32
                    or self._lib.BN_bn2binpad(output_y, y_buffer, 32) != 32
                ):
                    raise OpenSSLMSMUnavailable("BN_bn2binpad failed")
                x_value = int.from_bytes(x_buffer.raw, "big")
                y_value = int.from_bytes(y_buffer.raw, "big")
            from .bn254 import G1Point

            return G1Point(x_value, y_value, False)
        finally:
            # Cached points are owned by the accelerator and released in
            # ``close``.  Only points allocated after the bounded cache filled
            # are ephemeral and need per-call cleanup.
            for point in ephemeral_points:
                self._lib.EC_POINT_free(point)
            for scalar in scalars:
                self._lib.BN_free(scalar)
            if output_x:
                self._lib.BN_free(output_x)
            if output_y:
                self._lib.BN_free(output_y)
            if output:
                self._lib.EC_POINT_free(output)

    def scalar_mul(self, point: Any, scalar: int) -> Any:
        """Multiply one BN254 point without re-entering Python Jacobian code."""

        return self.msm(((point, int(scalar)),))

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            for point in getattr(self, "_point_cache", {}).values():
                self._lib.EC_POINT_free(point)
            getattr(self, "_point_cache", {}).clear()
            if getattr(self, "_group", None):
                self._lib.EC_GROUP_free(self._group)
                self._group = None
            if getattr(self, "_context", None):
                self._lib.BN_CTX_free(self._context)
                self._context = None


def _load() -> _OpenSSLMSM | None:
    try:
        return _OpenSSLMSM()
    except OpenSSLMSMUnavailable:
        return None


ACCELERATOR = _load()


def multi_scalar_mul(entries: Sequence[tuple[Any, int]]) -> Any | None:
    """Run one MSM natively, or return ``None`` when unavailable."""

    if ACCELERATOR is None:
        return None
    return ACCELERATOR.msm(entries)


__all__ = ["ACCELERATOR", "OpenSSLMSMUnavailable", "multi_scalar_mul"]
