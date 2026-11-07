//! The buffer the shader reports its findings in, and the analysis of those findings.

use crate::shader::{ELEMENT_BIT_SIZE, ELEMENT_SIZE, UNWRITTEN_BLOCK_MIN_WORDS, VEC_SIZE};
use std::{fmt, mem};

#[derive(Default)]
pub struct U64HexDebug(pub i64);

impl fmt::Debug for U64HexDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[derive(Copy, Clone)]
pub struct MostlyZeroArr<const LEN: usize>(pub [u32; LEN]);

impl<const LEN: usize> fmt::Display for MostlyZeroArr<LEN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if LEN < 8 {
            for i in 0..LEN {
                write!(f, "0x{:08X} ", self.0[i])?;
            }
            return Ok(());
        }
        for i in 0..LEN {
            let next_row = i / 16;
            if next_row > 0
                && self.0[next_row * 16..next_row * 16 + 16]
                    .iter()
                    .all(|v| v == &0u32)
            {
                continue; //skip entire line of zeroes
            }
            if i % 16 == 0 && i != 0 {
                write!(f, "   0x{:X}? ", i / 16)?;
            }
            let vali = self.0[i];
            if vali > 999999u32 {
                write!(f, "{:3}m", vali / 1000000u32)?;
            } else if vali > 9999u32 {
                write!(f, "{:3}k", vali / 1000u32)?;
            } else if vali > 0 {
                write!(f, "{:4}", vali)?;
            } else {
                write!(f, "    ")?; //zero
            }
            if i % 16 == 15 {
                writeln!(f)?;
            } else if i % 4 == 3 {
                write!(f, "|")?;
            } else if i % 4 == 1 {
                write!(f, " ")?;
            }
        }
        Ok(())
    }
}

impl<const LEN: usize> std::default::Default for MostlyZeroArr<LEN> {
    fn default() -> Self {
        Self([0; LEN])
    }
}

/// Shared between the CPU and the shader. Must have the same layout as `IOBuf` in the WGSL code.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct IOBuf {
    /// single-bit errors, by the index of the flipped bit
    pub err_bit1_idx: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    /// wrong values, by the count of wrong bits (32 wrong bits are counted at index 0)
    pub err_bitcount: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    /// wrong values except FFFFFFFF, by the count of one bits in the value read
    pub mem_bitcount: MostlyZeroArr<ELEMENT_BIT_SIZE>,
    pub actual_ff: u32,
    pub actual_max: u32,
    pub actual_min: u32,
    pub idx_max: u32,
    pub idx_min: u32,
    pub done_iter_or_err: u32,
    pub iter: u32,
    pub calc_param: u32,
    /// wrong values that look like the write never reached memory (marker or previous pass data)
    pub unwritten_count: u32,
    /// calc_param of the data held in this window before the last write
    pub prev_calc_param: u32,
    /// how the window's previous data (prev_calc_param) is checked: 0 - not checked
    /// (the window wasn't rewritten), 1 - previous data as is, 2 - previous data inverted
    pub stale_mode: u32,
    /// the pattern is written and expected XORed with this: 0 as is, FFFFFFFF inverted
    pub xor_mask: u32,
    pub first_elem: MostlyZeroArr<VEC_SIZE>,
}

/// The values a window is written with: the pattern for calc_param, as is or inverted
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pattern {
    pub calc_param: u32,
    pub inverted: bool,
}

impl Pattern {
    /// The initial fill of every window, kept by window 0 for the whole test
    pub fn initial() -> Self {
        Self {
            calc_param: IOBuf::calc_param_for(1, 0),
            inverted: false,
        }
    }

    /// Pattern written to window `window_idx` on iteration `iter` (from 1) of the main loop.
    /// Iterations come in pairs writing the same pattern as is, then inverted, so every bit
    /// of the memory holds both 0 and 1 and flips between them.
    pub fn for_iteration(iter: u32, window_idx: i64) -> Self {
        Self {
            calc_param: IOBuf::calc_param_for(iter.div_ceil(2), window_idx),
            inverted: iter.is_multiple_of(2),
        }
    }

    pub fn xor_mask(self) -> u32 {
        if self.inverted { u32::MAX } else { 0 }
    }

    /// Expected value of the u32 at `index` in the window, as computed by the shader
    #[allow(clippy::manual_rotate)] // this is kept as similar to WGSL shader as possible
    pub fn value(self, index: u32) -> u32 {
        // wrapping like u32 arithmetic in the shader (avoids overflow panics in debug builds)
        let addr = index.wrapping_add(self.calc_param).wrapping_add(1);
        let shift = addr % 31u32;
        let rotated = if shift > 0 {
            addr << shift | addr >> (32 - shift)
        } else {
            addr
        };
        rotated ^ self.xor_mask()
    }
}

// IOBuf must have the same layout as the WGSL struct: 3*32 + 12 u32 fields + vec4
const _: () = assert!(mem::size_of::<IOBuf>() == (3 * 32 + 12 + 4) * 4);

/// Why a window's wrong values look like writes that never reached memory
#[derive(Debug, PartialEq)]
pub enum UnwrittenEvidence {
    /// values still hold the pre-write marker or the data of the previous pass
    MarkerOrStale(i64),
    /// a big contiguous block reads entirely as zero
    ZeroBlock(i64),
}

impl UnwrittenEvidence {
    pub fn words(&self) -> i64 {
        match self {
            Self::MarkerOrStale(words) | Self::ZeroBlock(words) => *words,
        }
    }
    pub fn reason(&self) -> &'static str {
        match self {
            Self::MarkerOrStale(_) => "still hold the pre-write marker or the previous pass's data",
            Self::ZeroBlock(_) => "form one contiguous block reading exactly zero",
        }
    }
}

impl fmt::Display for IOBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.actual_min <= self.actual_max {
            write!(
                f,
                "values range: 0x{:08X}..=0x{:08X}",
                self.actual_min, self.actual_max
            )?;
        } else {
            // only FFFFFFFF-like wrong values were seen, so min/max were never updated
            write!(f, "values range: none (only FFFFFFFF-like)")?;
        }
        writeln!(
            f,
            "   FFFFFFFF-like count:{}    bit-level stats table:",
            self.actual_ff
        )?;
        writeln!(
            f,
            "         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF"
        )?;
        write!(f, "SinglIdx{}", self.err_bit1_idx)?;
        write!(f, "TogglCnt{}", self.err_bitcount)?;
        write!(f, "1sInValu{}", self.mem_bitcount)?;
        Ok(())
    }
}

impl IOBuf {
    pub fn for_initial_iteration() -> Self {
        let mut result = Self::default();
        result.prepare_next_iter_write();
        result
    }
    pub fn prepare_next_iter_write(&mut self) {
        self.reset_errors();
        self.iter += 1;
        self.set_calc_param_for_starting_window();
    }
    pub fn set_calc_param_for_starting_window(&mut self) {
        self.calc_param = Self::calc_param_for(self.iter, 0);
    }
    /// calc_param used to write window `window_idx` on iteration `iter` of the main loop
    pub fn calc_param_for(iter: u32, window_idx: i64) -> u32 {
        iter.wrapping_mul(0x100107)
            .wrapping_add(window_idx as u32 * 0x81_u32)
    }
    /// Sets the pattern the shader writes or expects
    pub fn set_pattern(&mut self, pattern: Pattern) {
        self.calc_param = pattern.calc_param;
        self.xor_mask = pattern.xor_mask();
    }
    /// Sets what the window held before its last write, None if it wasn't rewritten
    pub fn set_previous_pattern(&mut self, previous: Option<Pattern>) {
        match previous {
            None => self.stale_mode = 0,
            Some(previous) => {
                self.prev_calc_param = previous.calc_param;
                self.stale_mode = if previous.inverted { 2 } else { 1 };
            }
        }
    }
    /// Count of wrong 32-bit values found by the read pass
    pub fn wrong_words(&self) -> i64 {
        // wrong values reading as FFFFFFFF are counted only in actual_ff, not in mem_bitcount
        self.mem_bitcount.0.iter().map(|&v| v as i64).sum::<i64>() + self.actual_ff as i64
    }
    /// Recognizes errors caused by writes that never reached memory - typically a driver or
    /// tool problem. Only a big contiguous block counts: real memory faults (including unstable
    /// memory where single writes don't stick) give scattered errors instead, so these
    /// must stay reported as memory errors.
    pub fn unwritten_evidence(&self) -> Option<UnwrittenEvidence> {
        if self.idx_min > self.idx_max {
            return None; // no errors recorded
        }
        let range_words = self.idx_max as i64 - self.idx_min as i64 + 1;
        // practically every value in the error range is affected
        // (allowing a few values whose expected pattern equals the read one, so aren't errors)
        let is_contiguous_block = |words: i64| {
            range_words >= UNWRITTEN_BLOCK_MIN_WORDS && words * 1000 >= range_words * 999
        };
        let unwritten_words = self.unwritten_count as i64;
        if is_contiguous_block(unwritten_words) {
            return Some(UnwrittenEvidence::MarkerOrStale(unwritten_words));
        }
        let zero_words = self.mem_bitcount.0[0] as i64; // values with no bits set
        if is_contiguous_block(zero_words) {
            return Some(UnwrittenEvidence::ZeroBlock(zero_words));
        }
        None
    }
    pub fn reset_errors(&mut self) {
        *self = IOBuf {
            iter: self.iter,
            calc_param: self.calc_param,
            prev_calc_param: self.prev_calc_param,
            stale_mode: self.stale_mode,
            xor_mask: self.xor_mask,
            idx_max: u32::MIN,
            idx_min: u32::MAX,
            actual_max: u32::MIN,
            actual_min: u32::MAX,
            ..Self::default()
        };
    }
    pub fn get_error_addresses_and_count(
        &self,
        buf_offset: i64,
    ) -> Option<(std::ops::RangeInclusive<U64HexDebug>, i64)> {
        if self.done_iter_or_err == self.iter {
            None
        } else {
            Some((
                std::ops::RangeInclusive::<U64HexDebug>::new(
                    U64HexDebug(buf_offset + self.idx_min as i64 * ELEMENT_SIZE),
                    U64HexDebug(buf_offset + (self.idx_max as i64 + 1) * ELEMENT_SIZE - 1),
                ),
                self.wrong_words(),
            ))
        }
    }
    /// Checks the expected value the shader reported for vec4 1 against the CPU calculation
    pub fn check_vec_first(&self) -> Result<(), Box<dyn std::error::Error>> {
        const TEST_IDX: u32 = 1;
        let pattern = Pattern {
            calc_param: self.calc_param,
            inverted: self.xor_mask != 0,
        };
        let expected = pattern.value(TEST_IDX * VEC_SIZE as u32);
        if expected != self.first_elem.0[0] {
            println!("{} 0x{:08X}", self, expected);
            return Err("unexpected calculated value, maybe shader execution is broken".into());
        }
        Ok(())
    }
}

/// Memory errors accumulated over the whole run, used for the summary and the likely cause
#[derive(Default, Clone)]
pub struct ErrorTotals {
    /// wrong values that look like real memory errors
    pub memory_errors: i64,
    /// wrong values explained by writes that never reached memory
    pub unwritten: i64,
    /// read passes (over one window) that found memory errors
    pub error_passes: u64,
    single_bit_by_index: [u64; ELEMENT_BIT_SIZE],
    by_wrong_bit_count: [u64; ELEMENT_BIT_SIZE],
    all_ones: u64,
}

impl ErrorTotals {
    /// Adds the findings of one read pass. Returns the evidence of unwritten memory, if any.
    pub fn add(&mut self, buf: &IOBuf) -> Option<UnwrittenEvidence> {
        let wrong = buf.wrong_words();
        if wrong == 0 {
            return None;
        }
        let unwritten = buf.unwritten_evidence();
        let unwritten_words = unwritten.as_ref().map_or(0, UnwrittenEvidence::words);
        self.unwritten += unwritten_words.min(wrong);
        let memory_errors = wrong - unwritten_words.min(wrong);
        if memory_errors > 0 && unwritten.is_none() {
            // the bit-level stats describe memory errors only when the pass isn't dominated by
            // unwritten memory, otherwise they'd describe the missing writes
            self.memory_errors += memory_errors;
            self.error_passes += 1;
            for i in 0..ELEMENT_BIT_SIZE {
                self.single_bit_by_index[i] += buf.err_bit1_idx.0[i] as u64;
                self.by_wrong_bit_count[i] += buf.err_bitcount.0[i] as u64;
            }
            self.all_ones += buf.actual_ff as u64;
        } else if memory_errors > 0 {
            self.memory_errors += memory_errors;
            self.error_passes += 1;
        }
        unwritten
    }

    /// Plain-language description of what the error pattern is typical of
    pub fn likely_cause(&self) -> Option<String> {
        let wrong: u64 = self.by_wrong_bit_count.iter().sum();
        if wrong == 0 {
            return None;
        }
        let single_bit: u64 = self.single_bit_by_index.iter().sum();
        if single_bit * 10 >= wrong * 8 {
            let (top_bit, top_count) = self
                .single_bit_by_index
                .iter()
                .enumerate()
                .max_by_key(|&(_, count)| *count)
                .map(|(bit, &count)| (bit, count))
                .unwrap_or_default();
            if top_count * 10 >= single_bit * 8 && single_bit >= 3 {
                return Some(format!(
                    "single-bit errors concentrated on bit {top_bit} of 32-bit values - typical of one faulty \
                     data line or memory chip (hardware damage, a bad solder joint) rather than instability"
                ));
            }
            return Some(
                "scattered single-bit flips - typical of unstable memory: memory overclock or undervolt, \
                 overheating, or weak memory cells"
                    .to_owned(),
            );
        }
        if self.all_ones * 2 >= wrong {
            return Some(
                "values reading as FFFFFFFF - typical of memory or the memory controller not responding: \
                 severe instability (too high memory clock) or a hardware fault"
                    .to_owned(),
            );
        }
        Some(
            "whole values corrupted (many wrong bits at once) - typical of severe instability \
             (too high memory clock or too low voltage) or addressing faults"
                .to_owned(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// IOBuf as the read shader leaves it after finding errors in [idx_min, idx_max]
    fn with_errors(idx_min: u32, idx_max: u32) -> IOBuf {
        let mut buf = IOBuf::for_initial_iteration();
        buf.idx_min = idx_min;
        buf.idx_max = idx_max;
        buf.done_iter_or_err = 0xFFFF_FFFF;
        buf
    }

    #[test]
    fn no_errors_is_not_unwritten() {
        assert_eq!(IOBuf::for_initial_iteration().unwritten_evidence(), None);
    }

    #[test]
    fn block_of_marker_or_stale_values_is_unwritten() {
        // whole 2GB window still holding the marker, as with a skipped initial write
        let mut buf = with_errors(0, 0x1FFF_FFFF);
        buf.mem_bitcount.0[16] = 0x2000_0000;
        buf.unwritten_count = 0x2000_0000;
        assert_eq!(
            buf.unwritten_evidence(),
            Some(UnwrittenEvidence::MarkerOrStale(0x2000_0000))
        );
    }

    #[test]
    fn scattered_writes_not_sticking_are_real_errors() {
        // unstable memory where some writes don't stick leaves old data in scattered places:
        // that's a memory fault and must not be excused as a driver problem
        let mut buf = with_errors(0x100, 0x3BFF_FFFF);
        buf.mem_bitcount.0[16] = 0x500;
        buf.unwritten_count = 0x500;
        assert_eq!(buf.unwritten_evidence(), None);
    }

    #[test]
    fn whole_window_of_zeros_is_unwritten() {
        // Real failure on Intel Arc B580 with a 10.5GB single buffer exceeding maxBufferSize:
        // "Errors address range: 0xA0000000..=0xDFFFFFFF", every value read as 0x00000000
        let mut buf = with_errors(0xA000_0000 / 4, 0xDFFF_FFFF / 4);
        buf.mem_bitcount.0[0] = 0x1000_0000;
        assert_eq!(
            buf.unwritten_evidence(),
            Some(UnwrittenEvidence::ZeroBlock(0x1000_0000))
        );
        let mut totals = ErrorTotals::default();
        assert!(totals.add(&buf).is_some());
        assert_eq!(totals.memory_errors, 0);
        assert_eq!(totals.unwritten, 0x1000_0000);
    }

    #[test]
    fn scattered_bit_flips_are_real_errors() {
        // README example of real VRAM errors: few wrong values spread over a big range
        let mut buf = with_errors(0x100, 0x3BFF_FFFF);
        buf.mem_bitcount.0[15] = 0x1000;
        buf.mem_bitcount.0[16] = 0x2000;
        buf.mem_bitcount.0[0] = 3;
        assert_eq!(buf.unwritten_evidence(), None);
    }

    #[test]
    fn small_zero_area_is_a_real_error() {
        let mut buf = with_errors(0x1000, 0x1000 + 0x100 - 1);
        buf.mem_bitcount.0[0] = 0x100;
        assert_eq!(buf.unwritten_evidence(), None);
    }

    #[test]
    fn calc_param_matches_previous_scheme() {
        let mut buf = IOBuf::for_initial_iteration();
        for _ in 0..5 {
            buf.prepare_next_iter_write();
        }
        buf.set_calc_param_for_starting_window();
        assert_eq!(IOBuf::calc_param_for(buf.iter, 0), buf.calc_param);
        assert_eq!(
            IOBuf::calc_param_for(buf.iter, 3),
            buf.calc_param.wrapping_add(3 * 0x81)
        );
        // wraps like u32 arithmetic in the shader instead of panicking
        assert_eq!(
            IOBuf::calc_param_for(u32::MAX, 2),
            u32::MAX.wrapping_mul(0x100107).wrapping_add(2 * 0x81)
        );
    }

    #[test]
    fn iterations_pair_a_pattern_with_its_inverse() {
        let plain = Pattern::for_iteration(5, 2);
        let inverted = Pattern::for_iteration(6, 2);
        assert!(!plain.inverted && inverted.inverted);
        assert_eq!(plain.calc_param, inverted.calc_param);
        for index in [0, 1, 4, 1000, 0xFFFF_FFF0] {
            // every bit holds both 0 and 1 over the pair
            assert_eq!(inverted.value(index), !plain.value(index));
        }
        // the next pair writes different values
        assert_ne!(Pattern::for_iteration(7, 2).value(4), plain.value(4));
        // iteration 1 differs from the initial fill in every window but 0, so missing writes show
        assert_eq!(Pattern::for_iteration(1, 0), Pattern::initial());
        assert_ne!(
            Pattern::for_iteration(1, 1).value(4),
            Pattern::initial().value(4)
        );
    }

    #[test]
    fn shader_value_check_accounts_for_inversion() {
        for iteration in [1, 2] {
            let pattern = Pattern::for_iteration(iteration, 1);
            let mut buf = IOBuf::for_initial_iteration();
            buf.set_pattern(pattern);
            buf.first_elem.0[0] = pattern.value(4);
            assert!(buf.check_vec_first().is_ok());
            buf.first_elem.0[0] = !pattern.value(4);
            assert!(buf.check_vec_first().is_err());
        }
    }

    #[test]
    fn previous_pattern_sets_stale_mode() {
        let mut buf = IOBuf::for_initial_iteration();
        buf.set_previous_pattern(None);
        assert_eq!(buf.stale_mode, 0);
        buf.set_previous_pattern(Some(Pattern::for_iteration(3, 1)));
        assert_eq!(buf.stale_mode, 1);
        buf.set_previous_pattern(Some(Pattern::for_iteration(4, 1)));
        assert_eq!(buf.stale_mode, 2);
        assert_eq!(buf.prev_calc_param, Pattern::for_iteration(4, 1).calc_param);
    }

    fn totals_of(buf: &IOBuf) -> ErrorTotals {
        let mut totals = ErrorTotals::default();
        totals.add(buf);
        totals
    }

    #[test]
    fn single_faulty_bit_is_reported_as_hardware() {
        let mut buf = with_errors(0x100, 0x100_0000);
        buf.err_bit1_idx.0[13] = 40;
        buf.err_bitcount.0[1] = 40;
        buf.mem_bitcount.0[15] = 40;
        let cause = totals_of(&buf).likely_cause().unwrap();
        assert!(cause.contains("bit 13"), "{cause}");
    }

    #[test]
    fn spread_single_bit_flips_are_reported_as_instability() {
        let mut buf = with_errors(0x100, 0x100_0000);
        for bit in 0..32 {
            buf.err_bit1_idx.0[bit] = 2;
        }
        buf.err_bitcount.0[1] = 64;
        buf.mem_bitcount.0[16] = 64;
        let cause = totals_of(&buf).likely_cause().unwrap();
        assert!(cause.contains("unstable memory"), "{cause}");
    }

    #[test]
    fn all_ones_values_are_reported() {
        let mut buf = with_errors(0x100, 0x100_0000);
        buf.err_bitcount.0[16] = 100;
        buf.actual_ff = 100;
        let cause = totals_of(&buf).likely_cause().unwrap();
        assert!(cause.contains("FFFFFFFF"), "{cause}");
    }

    #[test]
    fn no_errors_has_no_cause() {
        assert_eq!(ErrorTotals::default().likely_cause(), None);
    }
}
