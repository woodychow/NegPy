import threading
from typing import Optional

import numpy as np
from negpy.kernel.image.logic import rgb_to_rgba_into
import wgpu  # type: ignore
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_DOWNSAMPLE_WGSL = """
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(dst_tex);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }
    let src_dims = textureDimensions(src_tex);
    let sx_scale = f32(src_dims.x) / f32(dst_dims.x);
    let sy_scale = f32(src_dims.y) / f32(dst_dims.y);

    let fx0 = f32(gid.x) * sx_scale;
    let fx1 = f32(gid.x + 1u) * sx_scale;
    let fy0 = f32(gid.y) * sy_scale;
    let fy1 = f32(gid.y + 1u) * sy_scale;

    let ix0 = max(0, i32(floor(fx0)));
    let ix1 = min(i32(src_dims.x) - 1, max(ix0, i32(ceil(fx1)) - 1));
    let iy0 = max(0, i32(floor(fy0)));
    let iy1 = min(i32(src_dims.y) - 1, max(iy0, i32(ceil(fy1)) - 1));

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var y = iy0; y <= iy1; y++) {
        let wy = max(0.0, min(f32(y + 1), fy1) - max(f32(y), fy0));
        for (var x = ix0; x <= ix1; x++) {
            let wx = max(0.0, min(f32(x + 1), fx1) - max(f32(x), fx0));
            let w = wx * wy;
            acc += textureLoad(src_tex, vec2<i32>(x, y), 0) * w;
            wsum += w;
        }
    }
    let out_col = select(textureLoad(src_tex, vec2<i32>(ix0, iy0), 0), acc / wsum, wsum > 0.0);
    textureStore(dst_tex, vec2<i32>(i32(gid.x), i32(gid.y)), out_col);
}
"""

_downsample_lock = threading.Lock()
_downsample_cache: dict = {}


def _get_downsample_resources(device, dst_w: int, dst_h: int):
    """Cached compute pipeline, scratch storage texture, and staging buffer per wgpu device."""
    if _downsample_cache.get("device") is not device:
        _downsample_cache.clear()
        shader = device.create_shader_module(code=_DOWNSAMPLE_WGSL)
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )
        _downsample_cache["device"] = device
        _downsample_cache["pipeline"] = pipeline
        _downsample_cache["bgl"] = pipeline.get_bind_group_layout(0)

    bytes_per_row = (dst_w * 16 + 255) & ~255
    req_size = bytes_per_row * dst_h
    if _downsample_cache.get("dims") != (dst_w, dst_h):
        old_tex = _downsample_cache.pop("tex", None)
        if old_tex is not None:
            try:
                old_tex.destroy()
            except Exception:
                pass
        tex = device.create_texture(
            size=(dst_w, dst_h, 1),
            format="rgba32float",
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )
        _downsample_cache["tex"] = tex
        _downsample_cache["view"] = tex.create_view()
        _downsample_cache["dims"] = (dst_w, dst_h)

    if _downsample_cache.get("staging_size", 0) < req_size:
        old_buf = _downsample_cache.pop("staging", None)
        if old_buf is not None:
            try:
                old_buf.destroy()
            except Exception:
                pass
        _downsample_cache["staging"] = device.create_buffer(
            size=req_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        _downsample_cache["staging_size"] = req_size

    return (
        _downsample_cache["pipeline"],
        _downsample_cache["bgl"],
        _downsample_cache["tex"],
        _downsample_cache["view"],
        _downsample_cache["staging"],
        bytes_per_row,
    )


def destroy_downsample_resources() -> None:
    """Release shared downsample texture and staging buffer before destroying the device."""
    with _downsample_lock:
        for key in ("tex", "staging"):
            res = _downsample_cache.pop(key, None)
            if res is not None:
                try:
                    res.destroy()
                except Exception:
                    pass
        _downsample_cache.clear()


class GPUTexture:
    """
    Hardware-backed texture wrapper.
    Defaults to rgba32float for high-dynamic-range processing.
    """

    def __init__(self, width: int, height: int, format: str = "rgba32float", usage: int = 0) -> None:
        self.width, self.height, self.format = width, height, format
        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("Hardware device required")

        if usage == 0:
            usage = (
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.COPY_SRC
            )

        self.texture = gpu.device.create_texture(size=(width, height, 1), format=format, usage=usage)
        self.view = self.texture.create_view()

        # Persistent staging buffers, allocated lazily and reused across calls
        self._readback_staging = None  # full-texture readback
        self._region_staging = None  # sub-region readback
        self._region_staging_size: int = 0  # allocated size of _region_staging
        self._rgba_scratch: Optional[np.ndarray] = None  # RGB -> RGBA padding buffer
        # Serializes readback against destroy: the UI thread probes textures (densitometer, pixel
        # readout) while the render worker may destroy them on file switch. Destroying a mapped
        # staging buffer panics in wgpu-native, and that panic is uncatchable.
        self._lock = threading.Lock()

    def upload(self, data: np.ndarray) -> None:
        """Transfers ndarray to VRAM."""
        gpu = GPUDevice.get()
        if not gpu.device:
            return

        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.shape[2] == 3:
            data = self._pad_to_rgba(data)

        gpu.device.queue.write_texture(
            {"texture": self.texture},
            data,
            {"bytes_per_row": data.shape[1] * 16, "rows_per_image": data.shape[0]},
            (data.shape[1], data.shape[0], 1),
        )

    def _pad_to_rgba(self, data: np.ndarray) -> np.ndarray:
        """RGB -> RGBA in a buffer kept across uploads of one shape."""
        shape = (data.shape[0], data.shape[1], 4)
        if self._rgba_scratch is None or self._rgba_scratch.shape != shape:
            self._rgba_scratch = np.empty(shape, dtype=np.float32)
        rgb_to_rgba_into(data, self._rgba_scratch)
        return self._rgba_scratch

    def readback_region(self, x: int, y: int, rw: int, rh: int) -> np.ndarray:
        """Downloads a sub-region from VRAM. Significantly faster than a full readback."""
        gpu = GPUDevice.get()
        with self._lock:
            if not gpu.device or not self.texture:
                return np.zeros((rh, rw, 4), dtype=np.float32)

            rw = min(rw, max(1, self.width - x))
            rh = min(rh, max(1, self.height - y))

            bytes_per_row = (rw * 16 + 255) & ~255
            required_size = bytes_per_row * rh

            if self._region_staging is None or self._region_staging_size < required_size:
                if self._region_staging is not None:
                    try:
                        self._region_staging.destroy()
                    except Exception as e:
                        logger.warning("Failed to destroy region staging buffer", exc_info=e)
                    try:
                        gpu.poll()
                    except Exception as e:
                        logger.warning("Failed to poll GPU device", exc_info=e)
                self._region_staging = None
                self._region_staging_size = 0
                try:
                    self._region_staging = gpu.device.create_buffer(
                        size=required_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
                    )
                    self._region_staging_size = required_size
                except Exception as e:
                    logger.warning("Failed to create region staging buffer", exc_info=e)
                    self._region_staging = None
                    self._region_staging_size = 0
                    raise
            staging = self._region_staging

            enc = gpu.device.create_command_encoder()
            enc.copy_texture_to_buffer(
                {"texture": self.texture, "origin": (x, y, 0)},
                {"buffer": staging, "bytes_per_row": bytes_per_row},
                (rw, rh, 1),
            )
            gpu.device.queue.submit([enc.finish()])

            try:
                staging.map_sync(mode=wgpu.MapMode.READ)
                view = staging.read_mapped()
                arr = np.frombuffer(view, dtype=np.float32).reshape((rh, bytes_per_row // 4))
                result = arr[:, : rw * 4].reshape((rh, rw, 4)).copy()
                staging.unmap()
                return result
            except Exception:
                self._region_staging = None
                self._region_staging_size = 0
                raise

    def readback(self) -> np.ndarray:
        """Downloads pixels from VRAM to CPU ndarray (float32)."""
        gpu = GPUDevice.get()
        with self._lock:
            if not gpu.device or not self.texture:
                return np.zeros((self.height, self.width, 4), dtype=np.float32)

            bytes_per_row = (self.width * 16 + 255) & ~255
            size = bytes_per_row * self.height

            if self._readback_staging is None:
                self._readback_staging = gpu.device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
            staging = self._readback_staging

            enc = gpu.device.create_command_encoder()
            enc.copy_texture_to_buffer(
                {"texture": self.texture},
                {"buffer": staging, "bytes_per_row": bytes_per_row},
                (self.width, self.height, 1),
            )
            gpu.device.queue.submit([enc.finish()])

            try:
                staging.map_sync(mode=wgpu.MapMode.READ)
                view = staging.read_mapped()
                arr = np.frombuffer(view, dtype=np.float32).reshape((self.height, bytes_per_row // 4))
                result = arr[:, : self.width * 4].reshape((self.height, self.width, 4)).copy()
                staging.unmap()
                return result
            except Exception:
                self._readback_staging = None
                raise

    def readback_downsampled(self, max_edge: int) -> np.ndarray:
        """Box-downsamples in VRAM before readback when larger than ``max_edge``."""
        if max_edge <= 0 or max(self.width, self.height) <= max_edge:
            return self.readback()

        gpu = GPUDevice.get()
        with self._lock:
            if not gpu.device or not self.texture or not self.view:
                return np.zeros((1, 1, 4), dtype=np.float32)

            scale = float(max_edge) / float(max(self.width, self.height))
            dst_w = max(1, round(self.width * scale))
            dst_h = max(1, round(self.height * scale))

            with _downsample_lock:
                pipeline, bgl, dst_tex, dst_view, staging, bytes_per_row = _get_downsample_resources(gpu.device, dst_w, dst_h)
                bind_group = gpu.device.create_bind_group(
                    layout=bgl,
                    entries=[
                        {"binding": 0, "resource": self.view},
                        {"binding": 1, "resource": dst_view},
                    ],
                )
                enc = gpu.device.create_command_encoder()
                cpass = enc.begin_compute_pass()
                cpass.set_pipeline(pipeline)
                cpass.set_bind_group(0, bind_group)
                cpass.dispatch_workgroups((dst_w + 7) // 8, (dst_h + 7) // 8, 1)
                cpass.end()
                enc.copy_texture_to_buffer(
                    {"texture": dst_tex},
                    {"buffer": staging, "bytes_per_row": bytes_per_row},
                    (dst_w, dst_h, 1),
                )
                gpu.device.queue.submit([enc.finish()])

                try:
                    staging.map_sync(mode=wgpu.MapMode.READ)
                    mapped = staging.read_mapped()
                    arr = np.frombuffer(mapped, dtype=np.float32).reshape((dst_h, bytes_per_row // 4))
                    result = arr[:, : dst_w * 4].reshape((dst_h, dst_w, 4)).copy()
                    staging.unmap()
                    return result
                except Exception:
                    _downsample_cache.pop("staging", None)
                    _downsample_cache["staging_size"] = 0
                    raise

    def destroy(self) -> None:
        """Forces hardware resource release."""
        with self._lock:
            try:
                self.view = None
                if self.texture:
                    self.texture.destroy()
                    self.texture = None
            except Exception as e:
                logger.warning("Failed to destroy GPU texture", exc_info=e)
            for attr in ("_readback_staging", "_region_staging"):
                buf = getattr(self, attr, None)
                if buf is not None:
                    try:
                        buf.destroy()
                    except Exception as e:
                        logger.warning(f"Failed to destroy GPU buffer ({attr})", exc_info=e)
                    setattr(self, attr, None)
            self._region_staging_size = 0
            self._rgba_scratch = None


class GPUBuffer:
    """Uniform or storage buffer wrapper."""

    def __init__(self, size: int, usage: int) -> None:
        gpu = GPUDevice.get()
        if not gpu.device:
            raise RuntimeError("Hardware device required")
        self.buffer = gpu.device.create_buffer(size=size, usage=usage)

    def upload(self, data: np.ndarray) -> None:
        gpu = GPUDevice.get()
        if not gpu.device:
            return
        gpu.device.queue.write_buffer(self.buffer, 0, data)

    def destroy(self) -> None:
        try:
            if self.buffer:
                self.buffer.destroy()
                self.buffer = None
        except Exception as e:
            logger.warning("Failed to destroy GPU buffer", exc_info=e)
