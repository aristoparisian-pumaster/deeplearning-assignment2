import os
import sys
import time
import ctypes
import re
from ctypes import wintypes

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('BaseAddress', wintypes.LPVOID),
        ('AllocationBase', wintypes.LPVOID),
        ('AllocationProtect', wintypes.DWORD),
        ('RegionSize', ctypes.c_size_t),
        ('State', wintypes.DWORD),
        ('Protect', wintypes.DWORD),
        ('Type', wintypes.DWORD),
    ]

PID = 17876

def get_process_epoch_logs(pid):
    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return None, False

    addr = 0
    mbi = MEMORY_BASIC_INFORMATION()
    MEM_COMMIT = 0x1000
    PAGE_READWRITE = 0x04
    PAGE_READONLY = 0x02

    pattern = re.compile(rb'Epoch \d+/\d+ \[.*?Val Acc: \d+\.\d+')
    matches = []

    while kernel32.VirtualQueryEx(handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
        if mbi.State == MEM_COMMIT and (mbi.Protect & (PAGE_READWRITE | PAGE_READONLY)):
            buf = ctypes.create_string_buffer(mbi.RegionSize)
            bytes_read = ctypes.c_size_t()
            if kernel32.ReadProcessMemory(handle, ctypes.c_void_p(mbi.BaseAddress), buf, mbi.RegionSize, ctypes.byref(bytes_read)):
                raw = buf.raw[:bytes_read.value]
                for m in pattern.findall(raw):
                    s = m.decode('ascii', errors='ignore')
                    if s not in matches:
                        matches.append(s)
        addr = (mbi.BaseAddress if mbi.BaseAddress else 0) + mbi.RegionSize

    kernel32.CloseHandle(handle)
    return matches, True

def main():
    print("=" * 70)
    print(f" LIVE TRAINING MONITOR FOR CNN MODELS (PID: {PID}) ")
    print("=" * 70)
    print("Tekan Ctrl+C untuk berhenti monitoring.\n")

    seen_count = 0
    while True:
        logs, is_running = get_process_epoch_logs(PID)
        if not is_running:
            print("\n[INFO] Proses training (PID 17876) telah selesai atau tidak ditemukan.")
            break

        if logs and len(logs) > seen_count:
            # Display new epochs
            for line in logs[seen_count:]:
                print(f"[{time.strftime('%H:%M:%S')}] {line}")
            seen_count = len(logs)
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring dihentikan.")
