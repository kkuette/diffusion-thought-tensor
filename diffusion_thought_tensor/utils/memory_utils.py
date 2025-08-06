"""
Memory monitoring utilities for DREAM-Enhanced Model Training
"""

import torch
import gc


class MemoryMonitor:
    """Enhanced memory monitoring for DREAM model"""

    def __init__(self, target_memory_gb=20):  # Higher limit for DREAM model
        self.target_memory_gb = target_memory_gb
        self.peak_memory_gb = 0
        self.memory_warnings = 0

    def get_current_memory_gb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0

    def get_max_memory_gb(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0

    def check_memory_usage(self, step=None):
        current_memory = self.get_current_memory_gb()
        max_memory = self.get_max_memory_gb()
        self.peak_memory_gb = max(self.peak_memory_gb, max_memory)

        if current_memory > self.target_memory_gb * 0.9:
            self.memory_warnings += 1
            print(f"⚠️  High memory usage: {current_memory:.1f}GB/{self.target_memory_gb}GB")
            return True
        return False

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        return self.get_current_memory_gb()

    def get_memory_summary(self):
        return {
            "current_memory_gb": self.get_current_memory_gb(),
            "max_memory_gb": self.get_max_memory_gb(),
            "peak_memory_gb": self.peak_memory_gb,
            "memory_warnings": self.memory_warnings,
        }