import unittest
import numpy as np
import wgpu  # type: ignore
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.infrastructure.gpu.resources import GPUTexture, GPUBuffer
from negpy.infrastructure.gpu.shader_loader import ShaderLoader


class TestGPUInfrastructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = GPUDevice.get()

    def test_device_singleton(self):
        """GPUDevice should be a singleton."""
        device2 = GPUDevice.get()
        self.assertIs(self.gpu, device2)

    def test_shader_loader(self):
        """ShaderLoader should load valid WGSL files."""
        import os

        # Test with a known shader file
        shader_path = os.path.join("src", "features", "geometry", "shaders", "transform.wgsl")
        if os.path.exists(shader_path):
            module = ShaderLoader.load(shader_path)
            self.assertIsNotNone(module)
            # Second load should return from cache
            module2 = ShaderLoader.load(shader_path)
            self.assertIs(module, module2)

    def test_shader_loader_follows_a_new_device(self):
        """A module compiled on a destroyed device is never handed to the next one."""
        import os

        if not self.gpu.is_available:
            self.skipTest("GPU not available")
        shader_path = os.path.join("negpy", "features", "geometry", "shaders", "transform.wgsl")
        before = ShaderLoader.load(shader_path)
        GPUDevice.destroy_singleton()
        device = GPUDevice.get().device
        after = ShaderLoader.load(shader_path)
        self.assertIsNot(before, after)
        device.create_compute_pipeline(layout="auto", compute={"module": after, "entry_point": "main"})

    def test_gpu_texture(self):
        """GPUTexture should initialize and upload data."""
        if not self.gpu.is_available:
            self.skipTest("GPU not available")

        tex = GPUTexture(100, 100)
        self.assertEqual(tex.width, 100)
        self.assertEqual(tex.height, 100)
        self.assertIsNotNone(tex.texture)
        self.assertIsNotNone(tex.view)

        # Test upload
        data = np.random.rand(100, 100, 3).astype(np.float32)
        tex.upload(data)

        # Test readback
        read_data = tex.readback()
        self.assertEqual(read_data.shape, (100, 100, 4))
        # Verify first 3 channels match uploaded data (with small precision tolerance)
        np.testing.assert_allclose(read_data[:, :, :3], data, atol=1e-5)
        # Verify alpha channel is 1.0 (as uploaded in upload method)
        np.testing.assert_allclose(read_data[:, :, 3], 1.0, atol=1e-5)

        # Test explicit destroy
        tex.destroy()
        self.assertIsNone(tex.texture)
        self.assertIsNone(tex.view)

    def test_gpu_texture_readback_downsampled(self):
        """GPUTexture.readback_downsampled area-averages in VRAM before reading back."""
        if not self.gpu.is_available:
            self.skipTest("GPU not available")

        data = np.empty((400, 600, 3), dtype=np.float32)
        checker = ((np.indices((400, 600)).sum(axis=0) % 2) * 0.6 + 0.2).astype(np.float32)
        data[:, :, 0] = checker
        data[:, :, 1] = 0.3
        data[:, :, 2] = 0.7

        tex = GPUTexture(600, 400)
        try:
            tex.upload(data)
            small = tex.readback_downsampled(150)
            self.assertEqual(small.shape, (100, 150, 4))
            np.testing.assert_allclose(small[:, :, 0], 0.5, atol=1e-4)
            np.testing.assert_allclose(small[:, :, 1], 0.3, atol=1e-4)
            np.testing.assert_allclose(small[:, :, 2], 0.7, atol=1e-4)
            np.testing.assert_allclose(small[:, :, 3], 1.0, atol=1e-4)
        finally:
            tex.destroy()

    def test_gpu_buffer(self):
        """GPUBuffer should initialize and upload data."""
        if not self.gpu.is_available:
            self.skipTest("GPU not available")

        buf = GPUBuffer(1024, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.assertIsNotNone(buf.buffer)
        self.assertEqual(buf.buffer.size, 1024)

        # Test upload
        data = np.zeros(256, dtype=np.float32)
        buf.upload(data)

        # Test explicit destroy
        buf.destroy()
        self.assertIsNone(buf.buffer)


if __name__ == "__main__":
    unittest.main()
