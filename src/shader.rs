//! The compute shader writing and checking the test pattern, and constants that must match it.

pub const GB: f32 = (1024 * 1024 * 1024) as f32;
pub const READ_SHADER: &[u32] = memtest_vulkan_build::compiled_vk_compute_spirv!(
    r#"
// Same layout as IOBuf on the CPU side. The counters the read shader updates from many
// invocations at once are atomic; atomic<u32> has the same size and alignment as u32.
struct IOBuf
{
    err_bit1_idx: array<atomic<u32>, 32>,
    err_bitcount: array<atomic<u32>, 32>,
    mem_bitcount: array<atomic<u32>, 32>,
    actual_ff: atomic<u32>,
    actual_max: atomic<u32>,
    actual_min: atomic<u32>,
    idx_max: atomic<u32>,
    idx_min: atomic<u32>,
    done_iter_or_err: atomic<u32>,
    iter: u32,
    calc_param: u32,
    unwritten_count: atomic<u32>,
    prev_calc_param: u32,
    stale_mode: u32,
    xor_mask: u32,
    first_elem: vec4<u32>
}

@group(0) @binding(0) var<storage, read_write> io: IOBuf;
@group(0) @binding(1) var<storage, read_write> test: array<vec4<u32>>;

// Must match WRITE_MARKER on the CPU side: windows are filled with it before their initial write
const WRITE_MARKER: u32 = 0xA5C396E1u;

fn addr_value_by_index(i:u32, calc_param:u32)->vec4<u32>
{
    let effective_index_of_u32 = i * 4u + calc_param;
    return vec4<u32>(effective_index_of_u32 + 1u, effective_index_of_u32 + 2u, effective_index_of_u32 + 3u, effective_index_of_u32 + 4u);
}

fn test_value_for_param(i:u32, calc_param:u32)->vec4<u32>
{
    let addrs : vec4<u32> = addr_value_by_index(i, calc_param);
    let shifts : vec4<u32> = addrs % 31u;
    // (32u - shifts) % 32u keeps the right shift below 32, since shifting by 32 is undefined in SPIR-V.
    // With shifts == 0 this yields addrs | addrs == addrs, matching check_vec_first on the CPU side.
    let rotated : vec4<u32> = (addrs << shifts) | (addrs >> ((32u - shifts) % 32u));
    return rotated;
}

fn splat(value:u32)->vec4<u32>
{
    return vec4<u32>(value, value, value, value);
}

// The pattern is written either as is (xor_mask 0) or bitwise inverted (xor_mask FFFFFFFF),
// so every bit of the memory holds both 0 and 1 over two iterations
fn test_value_by_index(i:u32)->vec4<u32>
{
    return test_value_for_param(i, io.calc_param) ^ splat(io.xor_mask);
}


const TEST_WINDOW_1D_MAX_GROUPS: u32 = 0x4000u;
const TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY: u32 = 0x2000u;//don't inner-multiply by window size

@compute @workgroup_size(64, 1, 1)
fn read(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    let addr_mod = effective_invocation_id % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let new_mod = (11 * effective_invocation_id + 999 * io.iter + io.calc_param +  7 * (effective_invocation_id / TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY)) % TEST_WINDOW_READ_ADDR_ROTATION_GRANULARITY;
    let effective_addr = effective_invocation_id - addr_mod + new_mod; //make read order a bit rotated, not strictly sequential
    let actual_value : vec4<u32> = test[effective_addr];
    let expected_value : vec4<u32> = test_value_by_index(effective_addr);
    if any(actual_value != expected_value) {
        //slow path, executed only on errors found
        // what this memory held before the last write: lets us tell "write never landed" from corruption
        // stale_mode: 0 - don't check, 1 - previous data as is, 2 - previous data inverted
        let stale_mask : u32 = select(0u, 0xFFFFFFFFu, io.stale_mode == 2u);
        let stale_value : vec4<u32> = test_value_for_param(effective_addr, io.prev_calc_param) ^ splat(stale_mask);
        for(var i: i32 = 0; i < 4; i++) {
            let actual_u32 = actual_value[i];
            let error_mask = actual_u32 ^ expected_value[i];
            if error_mask == 0 {
                continue;
            }
            if actual_u32 == WRITE_MARKER || (io.stale_mode != 0u && actual_u32 == stale_value[i]) {
                atomicAdd(&io.unwritten_count, 1u);
            }
            let one_bits = countOneBits(error_mask);
            if one_bits == 1
            {
                let bit_idx = firstLeadingBit(error_mask);
                atomicAdd(&io.err_bit1_idx[bit_idx], 1u);
            }
            atomicAdd(&io.err_bitcount[one_bits % 32u], 1u);
            let vec_addr: u32 = effective_addr * 4u + u32(i);
            atomicMax(&io.idx_max, vec_addr);
            atomicMin(&io.idx_min, vec_addr);
            atomicMax(&io.done_iter_or_err, 0xFFFFFFFFu); //ERROR_STATUS
            let actual_bits = countOneBits(actual_u32);
            if actual_bits == 32
            {
                atomicAdd(&io.actual_ff, 1u);
            }
            else
            {
                atomicAdd(&io.mem_bitcount[actual_bits], 1u);
                atomicMax(&io.actual_max, actual_u32);
                atomicMin(&io.actual_min, actual_u32);
            }
        }
    }
    //assign done_iter_or_err only on specific index (performance reasons)
    if effective_addr == 0 {
        atomicMax(&io.done_iter_or_err, io.iter);
    } else if effective_addr == 1 {
        io.first_elem = expected_value;
    }
}

@compute @workgroup_size(64, 1, 1)
fn write(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    //make global_invocation_id processing specific memory addr different on writing compared to reading
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 8u * TEST_WINDOW_1D_MAX_GROUPS;//don't inner-multiply by window size
    let proccessed_mod = effective_invocation_id % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = effective_invocation_id + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx);
}

@compute @workgroup_size(64, 1, 1)
fn emulate_write_bugs(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let effective_invocation_id: u32 = global_invocation_id[0] + global_invocation_id[1] * TEST_WINDOW_1D_MAX_GROUPS;
    let TEST_WINDOW_SIZE_GRANULARITY: u32 = 64u * 8u * TEST_WINDOW_1D_MAX_GROUPS;//don't inner-multiply by window size
    let proccessed_mod = effective_invocation_id % TEST_WINDOW_SIZE_GRANULARITY;
    let proccessed_idx = effective_invocation_id + TEST_WINDOW_SIZE_GRANULARITY - 2 * proccessed_mod - 1;
    test[proccessed_idx] = test_value_by_index(proccessed_idx);
    if proccessed_idx == 0xADBA {
        test[proccessed_idx][1] ^= 0x400000u;//error simulation for test
    }
}
"#
);

pub const WG_SIZE: i64 = 64;
pub const VEC_SIZE: usize = 4; //vector processed by single workgroup item
pub const ELEMENT_SIZE: i64 = std::mem::size_of::<u32>() as i64;
pub const ELEMENT_BIT_SIZE: usize = (ELEMENT_SIZE * 8) as usize;
pub const TEST_WINDOW_1D_MAX_GROUPS: i64 = 0x4000;
pub const TEST_WINDOW_SIZE_GRANULARITY: i64 =
    VEC_SIZE as i64 * WG_SIZE * ELEMENT_SIZE * TEST_WINDOW_1D_MAX_GROUPS * 8_i64;
pub const TEST_WINDOW_MAX_SIZE: i64 = 4 * 1024 * 1024 * 1024 - TEST_WINDOW_SIZE_GRANULARITY;
pub const TEST_DATA_KEEP_FREE: i64 = 400 * 1024 * 1024;
pub const MIN_WANTED_ALLOCATION: i64 = TEST_DATA_KEEP_FREE;
pub const ALLOCATION_TRY_STEP: i64 = TEST_DATA_KEEP_FREE;

pub const MAP_FOR_GETTING_ADDRESS_SIZE: u64 = 2 * 1024 * 1024; // signle huge page

/// Environment variable limiting a single test window size, in megabytes (rounded down to 128MB steps)
pub const MAX_WINDOW_MB_ENV: &str = "MEMTEST_VULKAN_MAX_WINDOW_MB";
/// Environment variable skipping one write pass, to check the "writes not applied" detection:
/// 0 skips the initial write of the last window, N>0 skips writing window 1 on iteration N
pub const EMULATE_SKIPPED_WRITE_ENV: &str = "MEMTEST_VULKAN_EMULATE_SKIPPED_WRITE_ITERATION";
/// Environment variable making allocations above this size (like 6G) fail as out of memory,
/// to check the retry with less memory without really running out of it
pub const EMULATE_ALLOCATION_LIMIT_ENV: &str = "MEMTEST_VULKAN_EMULATE_ALLOCATION_LIMIT";
/// Environment variable making a GPU pass fail, to check how failures are reported:
/// `device-lost`, `hang` or `panic`, optionally with the pass number like `hang:20` (default 1;
/// the first passes are the initial fill, two per window)
pub const EMULATE_FAILURE_ENV: &str = "MEMTEST_VULKAN_EMULATE_FAILURE";

/// Value every test window is filled with before its first write. The test pattern practically
/// never equals it, so reading it back means the write never reached memory. Must match the shader.
pub const WRITE_MARKER: u32 = 0xA5C3_96E1;
/// A contiguous block of at least this many u32 values that all read back as the marker,
/// the previous pass's data or exactly zero (the way out-of-bounds access behaves) is treated
/// as never written rather than as faulty VRAM, which gives scattered errors instead.
/// 0x40000 values = 1MB.
pub const UNWRITTEN_BLOCK_MIN_WORDS: i64 = 0x40000;
