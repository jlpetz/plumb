// Copyright 2026 the plumb Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Large/huge-page buffers and core topology for the multi-threaded DRAM regime.
//!
//! Allocation mirrors TMR-APP's non-stitched path (`memory/backend.rs`): one plain
//! `VirtualAlloc2(MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES)` with the NONPAGED_HUGE or
//! NONPAGED_LARGE attribute and the **matching alignment** (1 GiB / 2 MiB; the alignment is how
//! the page size is obtained, so it is never dropped or retried without). No placeholders and
//! no `MEM_REPLACE_PLACEHOLDER` anywhere: that path is the build-26100 bugcheck (0x139), and a
//! plain large-page commit + `VirtualFree(MEM_RELEASE)` is what TMR does on every run.

use std::ffi::c_void;
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE, LUID};
use windows_sys::Win32::Security::{
    AdjustTokenPrivileges, LUID_AND_ATTRIBUTES, LookupPrivilegeValueW, SE_LOCK_MEMORY_NAME,
    SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
};
use windows_sys::Win32::System::Memory::{
    MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_LARGE_PAGES, MEM_RELEASE,
    MEM_RESERVE, MemExtendedParameterAddressRequirements, MemExtendedParameterAttributeFlags,
    PAGE_READWRITE, VirtualAlloc2, VirtualFree,
};
use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformation, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

// winnt.h values (windows-sys has them under Win32_System_SystemServices).
const MEM_EXTENDED_PARAMETER_NONPAGED_LARGE: u64 = 0x08;
const MEM_EXTENDED_PARAMETER_NONPAGED_HUGE: u64 = 0x10;
const GIB: usize = 1 << 30;
const MIB2: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pages {
    Huge,
    Large,
    Small,
}

impl Pages {
    pub fn label(self) -> &'static str {
        match self {
            Pages::Huge => "1 GiB",
            Pages::Large => "2 MiB",
            Pages::Small => "4 KiB",
        }
    }
}

/// Enable SeLockMemoryPrivilege in this process's token. It must already be *assigned* to the
/// account (`tmr.exe --setup-large-pages`, elevated, then sign out and in).
pub fn enable_lock_memory_privilege() -> Result<(), String> {
    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: out-param is a local; the pseudo-handle needs no closing.
    if unsafe {
        OpenProcessToken(
            GetCurrentProcess(),
            TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
            &mut token,
        )
    } == 0
    {
        return Err(format!("OpenProcessToken failed ({})", unsafe {
            GetLastError()
        }));
    }
    let mut luid = LUID {
        LowPart: 0,
        HighPart: 0,
    };
    // SAFETY: constant wide string; out-param is a local.
    let ok = unsafe { LookupPrivilegeValueW(std::ptr::null(), SE_LOCK_MEMORY_NAME, &mut luid) };
    let result = if ok == 0 {
        Err(format!("LookupPrivilegeValueW failed ({})", unsafe {
            GetLastError()
        }))
    } else {
        let tp = TOKEN_PRIVILEGES {
            PrivilegeCount: 1,
            Privileges: [LUID_AND_ATTRIBUTES {
                Luid: luid,
                Attributes: SE_PRIVILEGE_ENABLED,
            }],
        };
        // SAFETY: `tp` is passed directly as an argument and outlives the call.
        let ok = unsafe {
            AdjustTokenPrivileges(token, 0, &tp, 0, std::ptr::null_mut(), std::ptr::null_mut())
        };
        // AdjustTokenPrivileges "succeeds" with ERROR_NOT_ALL_ASSIGNED (1300) when the account
        // doesn't hold the privilege, so GetLastError decides.
        let err = unsafe { GetLastError() };
        match (ok, err) {
            (0, e) => Err(format!("AdjustTokenPrivileges failed ({e})")),
            (_, 0) => Ok(()),
            (_, 1300) => Err("SeLockMemoryPrivilege is not assigned to this account \
                (run `tmr.exe --setup-large-pages` elevated, then sign out and back in)"
                .into()),
            (_, e) => Err(format!("AdjustTokenPrivileges: unexpected error {e}")),
        }
    };
    // SAFETY: token opened above.
    unsafe { CloseHandle(token) };
    result
}

/// One thread's buffer. Owned by exactly one worker at a time.
pub struct Region {
    ptr: *mut u64,
    /// Requested length in u64 (the allocation may be rounded up to the page size).
    len: usize,
    pub pages: Pages,
}

// SAFETY: a Region is a plain allocation; the harness hands each one to a single thread.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// Requested size in bytes (what the kernels touch and the throughput counts).
    pub fn bytes(&self) -> usize {
        self.len * 8
    }

    /// # Safety
    /// Only one thread may hold the returned slice at a time.
    #[allow(
        clippy::mut_from_ref,
        reason = "each region is driven by exactly one worker thread"
    )]
    pub unsafe fn slice(&self) -> &mut [u64] {
        // SAFETY: committed, zeroed allocation of at least `len` u64; exclusivity per contract.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: allocated by VirtualAlloc2 below as one reservation; plain release.
        unsafe { VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE) };
    }
}

fn attr_param(flags: u64) -> MEM_EXTENDED_PARAMETER {
    let mut p = MEM_EXTENDED_PARAMETER::default();
    p.Anonymous1._bitfield = MemExtendedParameterAttributeFlags as u64;
    p.Anonymous2.ULong64 = flags;
    p
}

/// Allocate `bytes` with exactly `pages`. Errors carry the Win32 code: 1450 = no contiguous
/// physical memory (fragmentation), 1314 = no SeLockMemoryPrivilege, 87 = a bad request (our bug).
pub fn alloc(bytes: usize, pages: Pages) -> Result<Region, u32> {
    let (page, attr) = match pages {
        Pages::Huge => (GIB, Some(MEM_EXTENDED_PARAMETER_NONPAGED_HUGE)),
        Pages::Large => (MIB2, Some(MEM_EXTENDED_PARAMETER_NONPAGED_LARGE)),
        Pages::Small => (4096, None),
    };
    let size = bytes.div_ceil(page) * page;
    // LIFETIME: `req` is stored in `params` as a raw pointer that the kernel dereferences inside
    // VirtualAlloc2, so it lives at function scope, past the call (an FFI pointer stored for a later call must outlive that call).
    let mut req = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: std::ptr::null_mut(),
        HighestEndingAddress: std::ptr::null_mut(),
        Alignment: page,
    };
    let mut params: Vec<MEM_EXTENDED_PARAMETER> = Vec::new();
    let mut flags = MEM_RESERVE | MEM_COMMIT;
    if let Some(a) = attr {
        flags |= MEM_LARGE_PAGES;
        params.push(attr_param(a));
        let mut p = MEM_EXTENDED_PARAMETER::default();
        p.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        p.Anonymous2.Pointer = &mut req as *mut _ as *mut c_void;
        params.push(p);
    }
    // SAFETY: `params` (and the `req` it points to) are live for the whole call.
    let ptr = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null(),
            size,
            flags,
            PAGE_READWRITE,
            if params.is_empty() {
                std::ptr::null_mut()
            } else {
                params.as_mut_ptr()
            },
            params.len() as u32,
        )
    };
    let _ = &req;
    if ptr.is_null() {
        return Err(unsafe { GetLastError() });
    }
    Ok(Region {
        ptr: ptr as *mut u64,
        len: bytes / 8,
        pages,
    })
}

/// Logical CPUs ordered physical-cores-first: the first thread of every core, then the second
/// (SMT siblings), so N threads use N distinct cores while N <= cores.
pub fn cpu_order() -> Vec<usize> {
    let mut len = 0u32;
    // SAFETY: size query with a null buffer.
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len) };
    let n = len as usize / size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
    let mut info = vec![SYSTEM_LOGICAL_PROCESSOR_INFORMATION::default(); n];
    // SAFETY: buffer sized from the query above.
    if unsafe { GetLogicalProcessorInformation(info.as_mut_ptr(), &mut len) } == 0 {
        return (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect();
    }
    let cores: Vec<Vec<usize>> = info
        .iter()
        .filter(|i| i.Relationship == RelationProcessorCore)
        .map(|i| {
            (0..usize::BITS as usize)
                .filter(|b| i.ProcessorMask >> b & 1 == 1)
                .collect()
        })
        .collect();
    let smt = cores.iter().map(Vec::len).max().unwrap_or(1);
    let mut order = Vec::new();
    for round in 0..smt {
        for core in &cores {
            if let Some(&cpu) = core.get(round) {
                order.push(cpu);
            }
        }
    }
    order
}

pub fn physical_cores() -> usize {
    let order = cpu_order();
    let mut len = 0u32;
    // SAFETY: size query only.
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len) };
    let n = len as usize / size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
    let mut info = vec![SYSTEM_LOGICAL_PROCESSOR_INFORMATION::default(); n];
    // SAFETY: buffer sized from the query above.
    if unsafe { GetLogicalProcessorInformation(info.as_mut_ptr(), &mut len) } == 0 {
        return order.len();
    }
    info.iter()
        .filter(|i| i.Relationship == RelationProcessorCore)
        .count()
}
