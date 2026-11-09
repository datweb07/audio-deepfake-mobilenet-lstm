"""Windows memory and disk measurements for the prospective CPU engine."""
import ctypes as ct
from ctypes import wintypes
import shutil


GIB = 1024 ** 3
MIN_C_FREE = 3 * GIB
MIN_COMMIT_FREE = 3 * GIB
MIN_RAM_FREE = 2 * GIB
MIN_D_FREE = 15 * GIB
WRITE_RESERVE = 128 * 1024 ** 2


class Performance(ct.Structure):
    _fields_ = [('cb', wintypes.DWORD)] + [
        (field, ct.c_size_t) for field in (
            'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
            'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')
    ] + [(field, wintypes.DWORD) for field in ('HandleCount', 'ProcessCount', 'ThreadCount')]


class ProcessMemory(ct.Structure):
    _fields_ = [(field, wintypes.DWORD) for field in ('cb', 'PageFaultCount')] + [
        (field, ct.c_size_t) for field in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def snapshot(root):
    system = Performance()
    process = ProcessMemory()
    system.cb = ct.sizeof(system)
    process.cb = ct.sizeof(process)
    psapi = ct.WinDLL('psapi', use_last_error=True)
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ct.POINTER(ProcessMemory), wintypes.DWORD]
    if not psapi.GetPerformanceInfo(ct.byref(system), system.cb):
        raise ct.WinError(ct.get_last_error())
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ct.byref(process), process.cb):
        raise ct.WinError(ct.get_last_error())
    return dict(
        c_free=shutil.disk_usage('C:/').free, d_free=shutil.disk_usage(root).free,
        ram_total=system.PhysicalTotal * system.PageSize,
        ram_available=system.PhysicalAvailable * system.PageSize,
        commit_limit=system.CommitLimit * system.PageSize,
        commit_used=system.CommitTotal * system.PageSize,
        commit_available=(system.CommitLimit - system.CommitTotal) * system.PageSize,
        process_rss=process.WorkingSetSize, process_peak_rss=process.PeakWorkingSetSize,
        process_private_commit=process.PagefileUsage,
    )


def pressure(measurement):
    if measurement['c_free'] < MIN_C_FREE or measurement['d_free'] < MIN_D_FREE + WRITE_RESERVE:
        return 'DISK_SPACE_PRESSURE'
    if measurement['commit_available'] < MIN_COMMIT_FREE:
        return 'SYSTEM_COMMIT_PRESSURE'
    if measurement['ram_available'] < MIN_RAM_FREE or measurement['process_rss'] > 4 * GIB:
        return 'PROCESS_RSS_PRESSURE'
    return None


class ResourceStop(RuntimeError):
    pass


def require_resources(root):
    measurement = snapshot(root)
    issue = pressure(measurement)
    if issue:
        raise ResourceStop(issue)
    return measurement
