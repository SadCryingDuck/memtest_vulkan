# Changelog

Changes in SadCryingDuck's fork of memtest_vulkan. Each release's notes on GitHub start with its section here.

## v0.13.0-duck

### Test types
- New `--test` option, and a Test setting on the start screen (shortcut `T`):
  - `thorough`: all of the test types below in turn, then a longer idle hold before checking all memory again, to find cells that slowly lose their data. 15 minutes by default. Use it to confirm a final overclock or to check a suspect card.
  - `bus`: 00000000/FFFFFFFF and 55555555/AAAAAAAA patterns that switch the most memory bus lines at once, which is where overclocked memory tends to fail.
  - `random`: all patterns in random access order, which stresses the memory timings rather than the bandwidth.
  - `pulsed`: bursts of load with idle gaps, like the uneven load of games.
  - `standard` is still the default and works as before.
- Error reports name the pattern and access order that found the error.

### Testing more of the VRAM
- The test memory is split into windows of 512MB to 1GB instead of up to 3.9GB. When the driver places part of the test in system RAM because other programs use the VRAM, only that window is dropped. Before, a big window could stay in the test, so part of the run checked system RAM instead of VRAM.
- Before testing a discrete GPU, the test shows how much of its memory other programs use, since that part can't be tested, and names the biggest users on Windows. With a second GPU, it suggests connecting the monitor to that one while testing, which frees the memory the desktop uses.

## v0.12.0-duck

- Only running out of memory is retried with less memory. A lost GPU, a hang or a crash now ends the test as a runtime error, where before the test could retry with less memory until a smaller size passed.
- A GPU hang is reported instead of freezing the program, and crashes are reported as runtime errors instead of "results invalid".
- GPUs without Vulkan 1.1 are skipped, and `--size` below the minimum stops with a clear message.
- Switched from the unmaintained erupt Vulkan bindings to ash.
- Licensed under GPL-3.0-or-later. `--version` shows the license, and releases include the license texts and signed build provenance.
