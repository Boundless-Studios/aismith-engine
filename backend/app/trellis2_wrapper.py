"""Python ctypes wrapper for the trellis2cpp shared library (pr-1 / ABI v11).

Matches trellis2_capi.h from the pr-1 branch which adds:
  - shape_enc + tex_dec + tex_flow + tex_flow_hr for full PBR texturing
  - t2_generate gains pipeline_type, background_mode, texture_steps
  - t2_mesh_has_pbr / t2_mesh_pbr for per-vertex colour/metallic/roughness
  - t2_bake_glb to produce a UV-atlas GLB in pure C++ (no CUDA)
"""

from __future__ import annotations

import ctypes
import os
from ctypes import c_int, c_char_p, c_void_p, c_uint64, c_float, c_uint8, POINTER
from pathlib import Path
from typing import Callable, Optional

# progress: (user, stage, step, total)
T2_PROGRESS_FN = ctypes.CFUNCTYPE(None, c_void_p, c_int, c_int, c_int)

# t2_pipeline_type enum values
T2_PIPE_AUTO   = 0
T2_PIPE_COARSE = 1
T2_PIPE_512    = 2
T2_PIPE_1024   = 3

# t2_background_mode enum values
T2_BACKGROUND_AUTO  = 0
T2_BACKGROUND_KEEP  = 1
T2_BACKGROUND_BLACK = 2
T2_BACKGROUND_WHITE = 3

# t2_caps bitmask
T2_CAP_COARSE  = 1
T2_CAP_512     = 2
T2_CAP_1024    = 4
T2_CAP_TEXTURE = 8

_EXPECTED_ABI = 11


class Trellis2Wrapper:
    def __init__(self, build_dir: Path):
        self.build_dir = build_dir
        bin_dir = build_dir / "bin" / "Release"
        rel_dir = build_dir / "Release"

        # Register DLL search paths so Windows can resolve ggml-*.dll
        if hasattr(os, "add_dll_directory"):
            for d in (bin_dir, rel_dir):
                if d.is_dir():
                    try:
                        os.add_dll_directory(str(d))
                    except Exception as e:
                        print(f"Warning: failed to add DLL directory {d}: {e}")
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                p = Path(path_dir.strip()) if path_dir.strip() else None
                if p and p.is_dir():
                    try:
                        os.add_dll_directory(str(p))
                    except Exception:
                        pass

        def find_lib(name: str) -> Path:
            for d in (bin_dir, rel_dir):
                p = d / name
                if p.is_file():
                    return p
            raise FileNotFoundError(
                f"Shared library {name!r} not found in {bin_dir} or {rel_dir}"
            )

        # Load ggml dependencies first, then the main trellis2 DLL
        try:
            self._lib_base = ctypes.CDLL(str(find_lib("ggml-base.dll")))
            self._lib_cpu  = ctypes.CDLL(str(find_lib("ggml-cpu.dll")))
            try:
                self._lib_cuda = ctypes.CDLL(str(find_lib("ggml-cuda.dll")))
            except FileNotFoundError:
                self._lib_cuda = None
            self.lib = ctypes.CDLL(str(find_lib("trellis2.dll")))
        except Exception as e:
            raise RuntimeError(f"Failed to load trellis2 shared libraries: {e}") from e

        self._bind()

    # ------------------------------------------------------------------
    # ABI binding
    # ------------------------------------------------------------------

    def _bind(self) -> None:
        lib = self.lib

        # int t2_abi_version(void)
        lib.t2_abi_version.argtypes = []
        lib.t2_abi_version.restype = c_int
        abi = lib.t2_abi_version()
        if abi != _EXPECTED_ABI:
            raise RuntimeError(
                f"trellis2 ABI version mismatch: expected {_EXPECTED_ABI}, got {abi}. "
                "Rebuild vendor/trellis2cpp/build from the pr-1 branch."
            )

        # t2_pipeline * t2_pipeline_load(dino, ss_flow, ss_dec,
        #     slat_flow, slat_hr_flow, shape_dec,
        #     shape_enc, tex_dec, tex_flow, tex_flow_hr,
        #     flags, err, err_len)
        lib.t2_pipeline_load.argtypes = [
            c_char_p,  # dino_gguf
            c_char_p,  # ss_flow_gguf
            c_char_p,  # ss_dec_gguf
            c_char_p,  # slat_flow_gguf
            c_char_p,  # slat_hr_flow_gguf
            c_char_p,  # shape_dec_gguf
            c_char_p,  # shape_enc_gguf   (texture, optional)
            c_char_p,  # tex_dec_gguf     (texture, optional)
            c_char_p,  # tex_flow_gguf    (texture, optional)
            c_char_p,  # tex_flow_hr_gguf (texture, optional)
            c_int,     # flags
            c_char_p,  # err buffer
            c_int,     # err_len
        ]
        lib.t2_pipeline_load.restype = c_void_p

        lib.t2_pipeline_free.argtypes = [c_void_p]
        lib.t2_pipeline_free.restype = None

        lib.t2_pipeline_backend.argtypes = [c_void_p]
        lib.t2_pipeline_backend.restype = c_char_p

        lib.t2_pipeline_is_fine.argtypes = [c_void_p]
        lib.t2_pipeline_is_fine.restype = c_int

        lib.t2_pipeline_caps.argtypes = [c_void_p]
        lib.t2_pipeline_caps.restype = c_int

        # t2_mesh_result * t2_generate(pipeline,
        #   image_bytes, image_len,
        #   pipeline_type, background_mode,
        #   seed, steps, guidance, texture_steps,
        #   progress, user,
        #   preview, preview_user,
        #   err, err_len)
        lib.t2_generate.argtypes = [
            c_void_p,       # pipeline
            c_void_p,       # image_bytes
            c_int,          # image_len
            c_int,          # pipeline_type (t2_pipeline_type)
            c_int,          # background_mode (t2_background_mode)
            c_uint64,       # seed
            c_int,          # steps  (<=0 -> default 12)
            c_float,        # guidance (< 0 -> default 7.5)
            c_int,          # texture_steps (<=0 -> default 12)
            T2_PROGRESS_FN, # progress callback (may be NULL)
            c_void_p,       # user pointer
            c_void_p,       # preview callback (NULL to disable)
            c_void_p,       # preview_user
            c_char_p,       # err buffer
            c_int,          # err_len
        ]
        lib.t2_generate.restype = c_void_p

        lib.t2_mesh_n_verts.argtypes  = [c_void_p]
        lib.t2_mesh_n_verts.restype   = c_int
        lib.t2_mesh_n_tris.argtypes   = [c_void_p]
        lib.t2_mesh_n_tris.restype    = c_int
        lib.t2_mesh_verts.argtypes    = [c_void_p]
        lib.t2_mesh_verts.restype     = POINTER(c_float)
        lib.t2_mesh_normals.argtypes  = [c_void_p]
        lib.t2_mesh_normals.restype   = POINTER(c_float)
        lib.t2_mesh_tris.argtypes     = [c_void_p]
        lib.t2_mesh_tris.restype      = POINTER(c_int)
        lib.t2_mesh_has_pbr.argtypes  = [c_void_p]
        lib.t2_mesh_has_pbr.restype   = c_int
        lib.t2_mesh_pbr.argtypes      = [c_void_p]
        lib.t2_mesh_pbr.restype       = POINTER(c_float)  # 6 * n_verts
        lib.t2_mesh_free.argtypes     = [c_void_p]
        lib.t2_mesh_free.restype      = None

        # t2_bake_glb: CPU xatlas UV-atlas bake -> GLB bytes
        lib.t2_bake_glb.argtypes = [
            POINTER(c_float),  # verts  (3*n_verts)
            c_int,             # n_verts
            POINTER(c_int),    # tris   (3*n_tris)
            c_int,             # n_tris
            POINTER(c_float),  # pbr    (6*n_verts, or NULL)
            c_int,             # texture_size (<=0 -> auto)
            c_int,             # component_filter (0=remove tiny, 1=largest, 2=all)
            POINTER(c_int),    # out_len
            c_char_p,          # err
            c_int,             # err_len
        ]
        lib.t2_bake_glb.restype = POINTER(c_uint8)

        lib.t2_free_buffer.argtypes = [POINTER(c_uint8)]
        lib.t2_free_buffer.restype  = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _enc(self, s: Optional[str]) -> Optional[bytes]:
        return s.encode("utf-8") if s else None

    def load_pipeline(
        self,
        dino: str,
        ss_flow: str,
        ss_dec: str,
        slat_flow: Optional[str] = None,
        slat_hr_flow: Optional[str] = None,
        shape_dec: Optional[str] = None,
        shape_enc: Optional[str] = None,
        tex_dec: Optional[str] = None,
        tex_flow: Optional[str] = None,
        tex_flow_hr: Optional[str] = None,
    ) -> c_void_p:
        err_buf = ctypes.create_string_buffer(512)
        pipeline = self.lib.t2_pipeline_load(
            self._enc(dino),
            self._enc(ss_flow),
            self._enc(ss_dec),
            self._enc(slat_flow),
            self._enc(slat_hr_flow),
            self._enc(shape_dec),
            self._enc(shape_enc),
            self._enc(tex_dec),
            self._enc(tex_flow),
            self._enc(tex_flow_hr),
            0,         # flags
            err_buf,
            len(err_buf),
        )
        if not pipeline:
            raise RuntimeError(
                f"t2_pipeline_load failed: {err_buf.value.decode('utf-8', errors='ignore')}"
            )
        return pipeline

    def free_pipeline(self, pipeline: c_void_p) -> None:
        self.lib.t2_pipeline_free(pipeline)

    def caps(self, pipeline: c_void_p) -> int:
        return self.lib.t2_pipeline_caps(pipeline)

    def has_texture(self, pipeline: c_void_p) -> bool:
        return bool(self.caps(pipeline) & T2_CAP_TEXTURE)

    def backend_name(self, pipeline: c_void_p) -> str:
        return self.lib.t2_pipeline_backend(pipeline).decode("utf-8")

    def generate(
        self,
        pipeline: c_void_p,
        image_bytes: bytes,
        seed: int = 0,
        pipeline_type: int = T2_PIPE_AUTO,
        background_mode: int = T2_BACKGROUND_AUTO,
        steps: int = 0,
        guidance: float = -1.0,
        texture_steps: int = 0,
        progress_cb: Optional[Callable[[int, int, int], None]] = None,
    ) -> dict:
        c_cb = None
        if progress_cb:
            def _inner(user, stage, step, total):
                try:
                    progress_cb(stage, step, total)
                except Exception as exc:
                    print(f"[trellis2_wrapper] progress callback error: {exc}")
            c_cb = T2_PROGRESS_FN(_inner)

        err_buf = ctypes.create_string_buffer(512)
        # Pass image bytes as a raw buffer
        img_buf = (ctypes.c_uint8 * len(image_bytes))(*image_bytes)

        mesh_res = self.lib.t2_generate(
            pipeline,
            img_buf,
            len(image_bytes),
            c_int(pipeline_type),
            c_int(background_mode),
            c_uint64(seed),
            c_int(steps),
            c_float(guidance),
            c_int(texture_steps),
            c_cb,
            None,   # user
            None,   # preview callback
            None,   # preview_user
            err_buf,
            len(err_buf),
        )
        if not mesh_res:
            raise RuntimeError(
                f"t2_generate failed: {err_buf.value.decode('utf-8', errors='ignore')}"
            )

        try:
            nv = self.lib.t2_mesh_n_verts(mesh_res)
            nt = self.lib.t2_mesh_n_tris(mesh_res)
            verts_ptr   = self.lib.t2_mesh_verts(mesh_res)
            normals_ptr = self.lib.t2_mesh_normals(mesh_res)
            tris_ptr    = self.lib.t2_mesh_tris(mesh_res)
            has_pbr     = bool(self.lib.t2_mesh_has_pbr(mesh_res))

            verts   = list(verts_ptr[: nv * 3])
            normals = list(normals_ptr[: nv * 3])
            tris    = list(tris_ptr[: nt * 3])
            pbr     = None

            if has_pbr:
                pbr_ptr = self.lib.t2_mesh_pbr(mesh_res)
                pbr = list(pbr_ptr[: nv * 6])  # (r,g,b, metallic, roughness, alpha) per vertex
        finally:
            self.lib.t2_mesh_free(mesh_res)

        return {
            "vertices": verts,
            "normals": normals,
            "faces": tris,
            "pbr": pbr,       # None if texture pipeline was not loaded
            "has_pbr": has_pbr,
        }

    def bake_glb(
        self,
        verts: list[float],
        tris: list[int],
        pbr: Optional[list[float]],
        texture_size: int = 0,
        component_filter: int = 0,
    ) -> bytes:
        """Bake a UV-atlas GLB entirely in C++ (xatlas unwrap + texel fill).

        Returns raw GLB bytes ready to write to disk.
        """
        n_verts = len(verts) // 3
        n_tris  = len(tris)  // 3

        c_verts = (c_float * len(verts))(*verts)
        c_tris  = (c_int   * len(tris))(*tris)
        c_pbr   = (c_float * len(pbr))(*pbr) if pbr else None

        out_len = c_int(0)
        err_buf = ctypes.create_string_buffer(512)

        buf_ptr = self.lib.t2_bake_glb(
            c_verts, n_verts,
            c_tris,  n_tris,
            c_pbr,
            texture_size,
            component_filter,
            ctypes.byref(out_len),
            err_buf,
            len(err_buf),
        )
        if not buf_ptr:
            raise RuntimeError(
                f"t2_bake_glb failed: {err_buf.value.decode('utf-8', errors='ignore')}"
            )
        try:
            glb_bytes = bytes(buf_ptr[: out_len.value])
        finally:
            self.lib.t2_free_buffer(buf_ptr)

        return glb_bytes
