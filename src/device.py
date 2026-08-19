"""
Device and Acceleration Hardware Detection Module.
Detects available CPU cores and GPU acceleration backends (CUDA, DirectML, MPS, OpenCL).
"""
import os
import platform
import subprocess
from typing import Dict, Any, Optional


def get_cpu_info() -> Dict[str, Any]:
    """Ritorna informazioni sulla CPU e sui core disponibili."""
    count = os.cpu_count() or 1
    return {
        "cores_logical": count,
        "cores_recommended": max(1, count - 1),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "system": f"{platform.system()} {platform.release()}"
    }


def detect_gpu() -> Dict[str, Any]:
    """
    Rileva schede video e framework di accelerazione GPU disponibili (CUDA, DirectML, MPS, OpenCL).
    """
    gpu_info = {
        "available": False,
        "backend": None,
        "device_name": None,
        "details": []
    }

    # 1. Check PyTorch CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["backend"] = "cuda"
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["details"].append(f"CUDA: {gpu_info['device_name']}")
            return gpu_info
    except ImportError:
        pass

    # 2. Check PyTorch DirectML (Intel / AMD / NVIDIA on Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            gpu_info["available"] = True
            gpu_info["backend"] = "directml"
            gpu_info["device_name"] = torch_directml.device_name(0)
            gpu_info["details"].append(f"DirectML: {gpu_info['device_name']}")
            return gpu_info
    except ImportError:
        pass

    # 3. Check PyTorch MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info["available"] = True
            gpu_info["backend"] = "mps"
            gpu_info["device_name"] = "Apple Silicon MPS"
            gpu_info["details"].append("Apple MPS")
            return gpu_info
    except ImportError:
        pass

    # 4. Check CuPy (CUDA)
    try:
        import cupy
        gpu_info["available"] = True
        gpu_info["backend"] = "cupy"
        gpu_info["device_name"] = cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()
        gpu_info["details"].append(f"CuPy CUDA: {gpu_info['device_name']}")
        return gpu_info
    except (ImportError, Exception):
        pass

    # 5. OS-level GPU discovery (Windows DirectX/WMI/CIM)
    if platform.system() == "Windows":
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                gpus = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
                if gpus:
                    gpu_info["device_name"] = ", ".join(gpus)
                    gpu_info["details"].append(f"Hardware OS: {gpu_info['device_name']}")
        except Exception:
            pass

    return gpu_info


def get_hardware_summary() -> str:
    """Restituisce una stringa formattata con il riepilogo dell'hardware di calcolo."""
    cpu = get_cpu_info()
    gpu = detect_gpu()

    gpu_status = f"{gpu['device_name']} (Backend: {gpu['backend']})" if gpu['available'] else (
        f"{gpu['device_name']} [Hardware presente, framework GPU non caricato]" if gpu['device_name'] else "Non rilevata"
    )

    summary = (
        f"=== COMPUTATIONAL HARDWARE SUMMARY ===\n"
        f"* OS: {cpu['system']} ({cpu['architecture']})\n"
        f"* CPU: {cpu['processor'] or 'Generic'} ({cpu['cores_logical']} core logici disponibili)\n"
        f"* CPU Parallel Workers consigliati: {cpu['cores_recommended']}\n"
        f"* GPU: {gpu_status}\n"
        f"======================================="
    )
    return summary


if __name__ == "__main__":
    print(get_hardware_summary())
