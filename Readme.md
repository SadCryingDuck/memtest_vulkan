# [memtest_vulkan](https://github.com/SadCryingDuck/memtest_vulkan) - GPU memory testing tool

This is SadCryingDuck's fork of [GpuZelenograd/memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan); its versions end with `-duck`. It adds command line options, a clear end to the standard test with a summary, protection against false errors from driver limits and memory placed in system RAM, an inverted test pattern and modernized tooling.

Opensource cross-platform tool written in vulkan compute to stress test video memory for stability during overclocking or repair.

Just start the application and wait for the standard 5-minute test to finish, or stop it earlier with Ctrl+C. Detected errors are displayed immediately during the test run, and a summary with the result is shown at the end.

Requires system-provided vulkan loader and driver supporting Vulkan 1.1 (already installed with graphics drivers on most OS).

## Installation & Usage (Windows)

### [**<kbd>📥</kbd>** Get .exe from latest release](https://github.com/SadCryingDuck/memtest_vulkan/releases/latest)
GitHub users also may want to try [CI build artifacts](https://github.com/SadCryingDuck/memtest_vulkan/actions) (only last 3 months kept).

Release files are built by GitHub Actions and come with a signed build record. To check that a download was built by this repository from its published source, use the [GitHub CLI](https://cli.github.com): `gh attestation verify memtest_vulkan-v0.12.0-duck-windows-x86_64.exe -R SadCryingDuck/memtest_vulkan`

Start test by double-clicking the utility, no installation / parameters / configuration / admin-rights required. The test starts by itself after a short countdown. Before that, you can type the number of the GPU to test, or press S (or an arrow key) to open the settings: the GPU, the [test](#test-types) (standard, or thorough for final checks), the duration (from a 1-minute check to an endless test), the amount of memory, and stopping the test at the first error, which is handy when tuning an overclock. Up/Down choose a setting and Left/Right change it, or type a number for a custom value: minutes for the duration, GB for the memory. G, T, D, M and E are shortcuts for the five settings. Enter starts the test, Esc quits.

While testing, a progress line shows the time, a progress bar, the speed and the errors found so far. At the first error the console beeps and its taskbar button flashes, so errors get noticed even in a minimized window. The summary at the end says what to do next.
![WindowsScreenshot](.github/memtest_vulkan_windows_rtx2070.png)

Any found errors are immediately reported with a multi-line details. Detailed descriptions given below may help in advanced cases, but most of the time it's enough just check if errors are absent or present
<a id="errors_screenshot">![ErrorsScreenshot](.github/memtest_vulkan_windows_rx580.png)</a>


<details><summary>
Alternatively, the tool can be started from cmdline
</summary>
<pre>
C:\gpu-tools\memory>memtest_vulkan.exe --device 1 --size 6G
memtest_vulkan v0.12.0-duck  https://github.com/SadCryingDuck/memtest_vulkan
Based on memtest_vulkan by GpuZelenograd. Press Ctrl+C to stop testing early, see --help for options.
Log file: C:\gpu-tools\memory\memtest_vulkan.log

1: Intel(R) Arc(TM) B580 Graphics  12GB discrete GPU  (DevId 0xE20B)
2: AMD Radeon(TM) Graphics  11GB integrated GPU  (DevId 0x13C0)
Testing Intel(R) Arc(TM) B580 Graphics  12GB discrete GPU  (DevId 0xE20B)
Test plan: 6.0GB as 6 windows of 1.000GB, duration 5:00 (standard test)
   0:00 / 5:00     written:    3.0GB  193.3GB/s  checked:    6.0GB  37.6GB/s  no errors
   0:01 / 5:00     written:   15.0GB  192.7GB/s  checked:   30.0GB  37.5GB/s  no errors
   0:06 / 5:00     written:   84.0GB  189.2GB/s  checked:  168.0GB  36.8GB/s  no errors
   0:36 / 5:00     written:  492.0GB  185.6GB/s  checked:  984.0GB  36.0GB/s  no errors
   1:06 / 5:00     written:  486.0GB  183.4GB/s  checked:  972.0GB  35.4GB/s  no errors
   1:36 / 5:00     written:  459.0GB  174.7GB/s  checked:  918.0GB  33.3GB/s  no errors
   2:06 / 5:00     written:  408.0GB  154.0GB/s  checked:  816.0GB  29.7GB/s  no errors
   2:36 / 5:00     written:  387.0GB  148.8GB/s  checked:  774.0GB  28.1GB/s  no errors
   3:06 / 5:00     written:  504.0GB  189.4GB/s  checked: 1008.0GB  36.8GB/s  no errors
   3:36 / 5:00     written:  501.0GB  187.9GB/s  checked: 1002.0GB  36.5GB/s  no errors
   4:06 / 5:00     written:  480.0GB  180.1GB/s  checked:  960.0GB  34.9GB/s  no errors
Pausing the GPU load for 15 seconds, then checking all memory again...
   4:45 / 5:00     written:  369.0GB  179.1GB/s  checked:  744.0GB  34.7GB/s  no errors
   5:00 / 5:00     written:  243.0GB  188.8GB/s  checked:  486.0GB  36.5GB/s  no errors
------------------------------------------------------------------------------
GPU:        Intel(R) Arc(TM) B580 Graphics  12GB discrete GPU  (DevId 0xE20B)
Tested:     6.0GB as 6 windows of 1.000GB, 1477 passes in 5:00
Throughput: written 4431.0GB at 176.8GB/s, checked 8868.0GB at 34.1GB/s
Result:     PASSED - no errors found
Next steps: Nothing to fix. When tuning an overclock, this setting passed: you can try the next
            step, and confirm the final setting with a longer test, like --duration 1h.
Log file:   C:\gpu-tools\memory\memtest_vulkan.log
------------------------------------------------------------------------------

memtest_vulkan: no errors, test PASSED.
  press any key to continue...
</pre>
</details>

### Command line options

All options are optional, `memtest_vulkan --help` lists them:

| Option | Meaning |
|---|---|
| `-d, --device <N>` | Test GPU number N as shown in the list, without the selection prompt |
| `-s, --size <SIZE>` | Memory to test, like `4G` or `512M` (default: all free VRAM) |
| `-t, --duration <TIME>` | How long to test, like `30s`, `10m`, `2h` or `endless` (default: `5m`, the standard test; `15m` for the thorough test) |
| `--test <TEST>` | What to test with: `standard` (default), `thorough`, `bus`, `random` or `pulsed`, see [Test types](#test-types) |
| `-w, --window-mb <MB>` | Limit one test window to MB megabytes (128MB steps), to tell driver problems from memory errors |
| `--log <FILE>`, `--no-log` | Log file location (default: `memtest_vulkan.log` in the current folder), or no log |
| `-e, --stop-on-error` | End the test at the first error, like when tuning an overclock |
| `-l, --list` | List the GPUs and exit |
| `-b, --batch` | No prompts and no key press at the end, for scripts |
| `-v, --verbose` | Print diagnostic information (renaming the executable to `memtest_vulkan_verbose` works too) |

Press Ctrl+C once to stop testing and show the results, twice to quit at once. A test stopped after the standard 5 minutes still counts as complete (the thorough test needs its whole duration). With `--device` or `--batch` the start screen is skipped, so scripts start testing at once.

The exit code tells the result, which is useful in scripts (`memtest_vulkan --batch --duration 1h`):

| Exit code | Meaning |
|---|---|
| 0 | test passed, no errors |
| 1 | memory errors found |
| 2 | results invalid: test writes didn't reach the memory (likely a driver problem, see [Troubleshooting](#troubleshooting)) |
| 3 | the test couldn't run, or stopped because of a runtime error |
| 4 | stopped before the test duration ended (or cancelled at the start screen), no errors so far |

The older form `memtest_vulkan <device> [<bytes>]` still works and tests without a time limit.

### Installation & Usage (Linux)

Install by unpacking archives with linux prebuilt binaries for X86_64 (Desktop) or AARCH64 (Embedded) from 
[Releases](https://github.com/SadCryingDuck/memtest_vulkan/releases/latest) or [![ci build](https://github.com/SadCryingDuck/memtest_vulkan/actions/workflows/ci.yml/badge.svg) artifacts](https://github.com/SadCryingDuck/memtest_vulkan/actions) (github account required). Build artifacts are transparently produced from source by github actions, but only last 3 months kept.

Use by opening a terminal in a folder with extracted file and explicitly running `./memtest_vulkan`. Do NOT just double-click binary in GUI (it would lead to starting test in the background without ability to stop it - [issue](https://github.com/GpuZelenograd/memtest_vulkan/issues/11))

Linux platform often contains additional `llvmpipe` pure-CPU vulkan driver. So after the start device selection menu will be shown. You can wait 10 seconds for automatic device selection or manually type the device number to test
![LinuxScreenshot](.github/linux-laptop-igpu.png)


<details><summary>With multiple drivers packages installed running under linux may require explicitly setting environment variables
</summary>
The details & reasons about the variables are discussed in [troubleshooting section](#troubleshooting), here are only execution examples
<pre>
[user@host ~]$ VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x01:00 DevId=0x2204   24GB NVIDIA GeForce RTX 3090
Testing 1: Bus=0x01:00 DevId=0x2204   24GB NVIDIA GeForce RTX 3090
      1 iteration. Since last report passed 56.112854ms     written    19.5GB, read:    22.8GB    752.9GB/sec
     19 iteration. Since last report passed 1.011701765s    written   351.0GB, read:   409.5GB    751.7GB/sec
    199 iteration. Since last report passed 10.050222094s   written  3510.0GB, read:  4095.0GB    756.7GB/sec
   1954 iteration. Since last report passed 100.004113065s  written 34222.5GB, read: 39926.2GB    741.5GB/sec
^C
memtest_vulkan: no any errors, testing PASSed.
  press any key to continue...
</pre>

Example run with a single-wire/singe-bit error
<pre>
[user@host ~]$ ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x01:00 DevId=0x1B87   8GB NVIDIA P104-100
Testing 1: Bus=0x01:00 DevId=0x1B87   8GB NVIDIA P104-100
      1 iteration. Since last report passed 52.20479ms      written     3.8GB, read:     7.5GB    215.5GB/sec
     21 iteration. Since last report passed 1.0515038s      written    75.0GB, read:   150.0GB    214.0GB/sec
    216 iteration. Since last report passed 10.021230569s   written   731.2GB, read:  1462.5GB    218.9GB/sec
   2125 iteration. Since last report passed 100.010942973s  written  7158.8GB, read: 14317.5GB    214.7GB/sec
Error found. Mode NEXT_RE_READ, total errors 0x3C7EC3 out of 0x3C000000 (0.39384872%)
Errors address range: 0x9D66148C..=0xDCD3036B  deatils:
         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF
Err1BIdx                 |      1m         |                 |                 
   0x1?                  |      1m         |                 |                 
ErrBiCnt      3m 820k    |                 |                 |                 
MemBiCnt            1   2|  32 249 13645067| 15k 39k  81k142k|219k308k 398k468k
   0x1? 506k502k 448k353k|239k134k  63k 25k|79792113  310  43|   5        1    
actual_ff: 0 actual_max: 0xFFFFFFB7 actual_min: 0x00000730 done_iter_or_err:4294967295 iter:1 calc_param 0x00100107
Error found. Mode INITIAL_READ, total errors 0x7E0C6E out of 0x3C000000 (0.82062860%)
Errors address range: 0x11640B6C4..=0x1DFFFEFFF  deatils:
         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF
Err1BIdx                 |      3m         |                 |                 
   0x1?                  |      3m         |                 |                 
ErrBiCnt      6m   1m    |                 |      51    3 598|  302573   824924
   0x1?  1084402   772471|  22 878    7 152|   1   4    1    |                 
MemBiCnt                7|  43 285 15296317| 19k 50k 107k200k|326k483k 653k817k
   0x1? 949k  1m 999k886k|704k493k 297k149k| 62k 20k 57931263| 185  21    1    
actual_ff: 0 actual_max: 0xFFFFFF46 actual_min: 0x000000B0 done_iter_or_err:4294967295 iter:2160 calc_param 0x8708AB91
Runtime error: ERROR_DEVICE_LOST while getting () in context wait_for_fences
</pre>
...hangs in-kernel due to driver
</details>

<details><summary>
64-bit ARM platforms are supported by AARCH64 binary.
</summary>

Example run on NVIDIA Jetson
<pre>
jetson-nx-alpha :: ~ » ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x00:00 DevId=0xA5BA03D7   8GB NVIDIA Tegra Xavier (nvgpu)
Testing 1: Bus=0x00:00 DevId=0xA5BA03D7   8GB NVIDIA Tegra Xavier (nvgpu)
      1 iteration. Since last report passed 163.678336ms    written     2.4GB, read:     4.8GB     43.5GB/sec
      7 iteration. Since last report passed 1.045756448s    written    14.2GB, read:    28.5GB     40.9GB/sec
     61 iteration. Since last report passed 10.06722992s    written   128.2GB, read:   256.5GB     38.2GB/sec
    593 iteration. Since last report passed 100.063183744s  written  1263.5GB, read:  2527.0GB     37.9GB/sec
   1121 iteration. Since last report passed 100.043447136s  written  1254.0GB, read:  2508.0GB     37.6GB/sec
^C
memtest_vulkan: no any errors, testing PASSed.
  press any key to continue...
</pre>

Raspberry 4 with 64-bit Broadcom V3D vulkan driver is also supported, but the achieved performance is quite low. No GUI required, test can be started via SSH connection:
<pre>
root@raspberrypi:/root# ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.4.2 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x00:00 DevId=0xBE485FD3   1GB V3D 4.2
2: Bus=0x00:00 DevId=0x0000   2GB llvmpipe (LLVM 14.0.6, 128 bits)
(first device will be autoselected in 0 seconds)   Override index to test:
    ...first device autoselected
Standard 5-minute test of 1: Bus=0x00:00 DevId=0xBE485FD3   1GB V3D 4.2
      1 iteration. Passed  1.8044 seconds  written:    0.2GB   0.6GB/sec        checked:    0.5GB   0.4GB/sec
      2 iteration. Passed  1.8037 seconds  written:    0.2GB   0.6GB/sec        checked:    0.5GB   0.4GB/sec
      5 iteration. Passed  5.4135 seconds  written:    0.8GB   0.6GB/sec        checked:    1.5GB   0.4GB/sec
     22 iteration. Passed 30.6776 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
     39 iteration. Passed 30.6716 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
     56 iteration. Passed 30.6738 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
     73 iteration. Passed 30.6744 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
     90 iteration. Passed 30.6714 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
    107 iteration. Passed 30.6736 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
    124 iteration. Passed 30.6688 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
    141 iteration. Passed 30.6810 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
    158 iteration. Passed 30.6751 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
Standard 5-minute test PASSed! Just press Ctrl+C unless you plan long test run.
Extended endless test started; testing more than 2 hours is usually unneeded
use Ctrl+C to stop it when you decide it's enough
    192 iteration. Passed 30.6733 seconds  written:    4.2GB   0.6GB/sec        checked:    8.5GB   0.4GB/sec
^C
memtest_vulkan: no any errors, testing PASSed.
  press any key to continue...
</pre></details>

### Test types

Different memory faults show up under different conditions, so besides the standard test there are tests that stress the memory in other ways:

| Test | What it does | Good for |
|---|---|---|
| `standard` | Values derived from each address, then the same inverted, written and read at full speed | Most problems: a quick, reliable check |
| `bus` | 00000000/FFFFFFFF and 55555555/AAAAAAAA alternating word by word, so the most bits switch at once on the memory bus | Overclocked memory, which tends to fail when many bus lines switch together |
| `random` | All patterns, going through memory in random order, so nearly every access opens another memory row | The memory timings, rather than the bandwidth |
| `pulsed` | All patterns in bursts of load from 0.1 to 2 seconds with idle gaps between them | Errors from the power and clock changes of an uneven load, like in games |
| `thorough` | All of the above in turn, then a longer idle hold (10% of the time, 30 seconds to 5 minutes) before checking all memory again | Final checks of an overclock, and suspect cards: the hold finds cells that slowly lose their data |

Every test writes each pattern as is and then inverted, so every bit holds both 0 and 1, and keeps the window that's never rewritten to check that data stays intact. The thorough test lasts 15 minutes unless a duration is given. The memory bus layout and the data bus inversion of GDDR memory differ between cards, so no pattern is guaranteed to be the worst case for a particular card: that's why the thorough test uses several.

# Interpreting the results

The _only_ simple thing in result interpreting - is "If some errors are reported by memtest_vulkan - the PC has hardware problems". Most often GPU-related, but its hard to be sure if the problem is VRAM IC - related or GPU chip related.

The summary at the end of the test also gives the *likely cause* the error pattern is typical of: single-bit flips concentrated on one bit point to one faulty data line or chip, scattered single-bit flips to unstable memory (overclock, undervolt, overheating), values reading as FFFFFFFF or whole corrupted values to severe instability. These are hints based on typical patterns, not a diagnosis.

The test can only reach the video memory no other program uses. On a discrete GPU it shows how much other programs use when that's more than 0.3GB, naming the biggest users on Windows. Closing them lets the test cover more of the memory. On a PC with a second GPU (like the one built into the CPU), connecting the monitor to that one while testing also frees the memory the Windows desktop uses.

The test memory is split into windows of 512MB to 1GB. If a test window reads many times slower than the others, the driver placed that part of the test memory in system RAM, usually because other programs use the VRAM. The test then drops that window and continues on the rest, so it keeps checking VRAM only; the summary shows how much was skipped. (The window holding the re-read data is never dropped: if it's the slow one, the test only warns that part of it checks system RAM.)

Integrated GPUs use system RAM as video memory, so by default they test only the system RAM that is available, minus a reserve for the system and other programs (20% of the total RAM, at least 2GB). The amount is shown when the test starts, and `--size` changes it.

You can create [topic in discussions](https://github.com/GpuZelenograd/memtest_vulkan/discussions/new?category=card-specific-memtest_vulkan-error-logs), sometimes the error log can be interpreted. Bust most of the time it is quite complex task, expand the theory section below

<details><summary>
      
## Video card memory error kinds (theory)

</summary>

### Classification of the errors by the "*what* have gone wrong"
* The single-bit errors like in an [image above](#errors_screenshot). Such errors are counted in ToggleCnt column 0x01 and the exact bit indices are counted in SingleIdx column. Such errors may be detected by EDC in theory if they occur during transmitting by EDC-enabled part of GPU<->memory wire. But I'm not sure if EDC helps if they occure when transmitting between gpu cache and gpu core or something like this.
* The errors on data-inversion bit (if not detected by EDC). Those should be counted in ToggleCnt columns 0x07/0x08 without SingleIdx info for them.
* The multi-bit transmission errors. Those should be counted in ToggleCnt columns above 0x01, without SingleIdx info for them.
* The errors flipped in the memory chips itself during data storage/"refresh cycles". This may be caused by too big period of refresh or other problems. memtest_vulkan uses a part of memory in a "write once at start but reread every time" pattern - it is the reason fot read GB is more then written GB. If a data flips inside this part of memory - there would be infinite log of error messages marked with "Mode NEXT_RE_READ" (in opposite to Mode INITIAL_READ). Lowering the clocks without restarting test doesn't get rid of such errors.
* Stuck bits and interference between neighbouring cells: a bit that can't change, or that changes when nearby bits are written. The iterations come in pairs: the first writes the test pattern as is, the second writes the same pattern bitwise inverted, so every bit is checked holding both 0 and 1. A stuck bit gives errors on every second iteration, showing in the SinglIdx column at the same index. (The part of memory used for the re-read check keeps its initial data, so it isn't inverted.)
* The errors on the address-transmission bus. The metest_vulkan is designed to perform reads to the non-sequential series of medium-sized sequential blocks. And if the address is wrongly interpreted by a memory chip - the result is completely garbage from wrong cell. Data-bus EDC can't help here. Those errors typically gives completely random error patterns with normal distribution of bits count and flipped bits (so typical number of flipped bits are 12-20 of 32 and getting 1 flipped bit for this case is extremely unrealistic). The result looks like
```
Error found. Mode INITIAL_READ, total errors 0x2B788 out of 0x18000000 (0.04422069%)
Errors address range: 0x6000E900..=0xBFDFF9FF  iteration:38
values range: 0xFFFFA1A4..=0x0000166F   FFFFFFFF-like count:0    bit-level stats table:
         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF
SinglIdx                 |                 |                 |                 
TogglCnt                2|   7  18   95 264| 8451786 40056770| 11k 15k  20k 23k
   0x1?  23k 21k  17k 12k|81944859 24701266| 486 248   62  29|   4   2         
1sInValu                3|  19  66  223 700|17683704 6856 11k| 16k 21k  25k 26k
   0x1?  23k 17k  12k6327|2883 917  282  64|   9             |
```
* Other critical errors inside memory chips or memory controller. This gives normal distributions for TogglCnt, but for 1sInValu the distribution may be different - since critical internal errors may be reported by some fixed patterns (0x00000000, 0xFFFFFFFF - for some EDC problems, 0x0BADAC?? - for some NVIDIA problems).
* Memory errors in the areas where error counts are stored)) This often shows as millions of errors in all table entries, typically with the total errors greater than tested memory size. Such results are numerically garbage, but means that the gpu/memory is really mostly non-functional.
* The errors in GPU during calculation of addresses and desired values or in value comparison. This can lead to any pattern of reporting at all, since the logic of a program is broken.


### Orthogonal classification by "*when* things have gone wrong"
* Simplest situation: "errors appears immediately when GPU+memory operates at given frequency". The tests report such nearly immediately.
* Temperature-dependent: sometimes the system works fine at room temperature but after getting hot due to continuous load the errors are coming. To catch such errors some pre-heat time is needed, the 5-6 minutes of standard test are designed exactly to wait for achieving higher temperatures.
* Near-the-limit, rare-occurring errors. When hardware is working near its limits, the errors can present but be fairly rare or depend on the outer factors like "electrical power network noise caused by powering on a drill in a nearby office". Catching such errors may require 2-3 hours of test run.
* Lower-frequency errors. Sometimes faulty memory or GPU can handle high-performance "huge frequency and medium timings" mode but fails to handle the low-performance "low frequency and small timings" mode. Such errors are hard to catch; the standard 5-6 minute test sometimes catches them during a special "several seconds pause of load" at the beginning of the 5-th minute, but that is not a reliable method.
* Frequency-switch errors. The GPU+memory may work fine at both high- and low-performance modes, but it may fail at the moment of the switching while clocks and timings are adjusted. Similar to above, hard to catch, sometimes caught at the beginning of the 5th minute.
Testing mode for the two later categories is still under development - it's tricky to check memory operation at lower frequency, since the driver nearly immediate switches to performance mode when load begins. However, v0.5 introduces a preliminary attempt to handle this. After initial pre-heat, 30 seconds before the end of the test (of the standard 5 minutes for longer tests), the load is stopped for 15 seconds, then all the memory is checked again without rewriting it (reported as Mode AFTER_PAUSE) and the load raises again.
</details>

# <a id="troubleshooting">Troubleshooting & reporting bugs</a>
Here is the list of common errors that prevent test from starting
* `memtest_vulkan couldn't start testing: ... failed to load`<br>
This message means that your system lacks the Khronos Group Vulkan-Loader library. This library is used as a multiplexer between different drivers provided for different devices and typically is installed during installation of any device-specific vulkan driver. However, some platforms may need explicit installation: for example, to install it on ubuntu 18.04 run `sudo apt install libvulkan1`.
Note that this library itself doesn't depend on any GPU, it is loadable even without any vulkan-capable devices at all. So the error above is a pure software-related error, not related to hardware at all. For windows7 x64 you may need [downloading `vulkan-1.dll` manually](https://github.com/GpuZelenograd/memtest_vulkan/releases/tag/support).
* `memtest_vulkan couldn't start testing: ERROR_INCOMPATIBLE_DRIVER`<br>
`memtest_vulkan couldn't start testing: ERROR_INITIALIZATION_FAILED`<br>
Those messages mean that your system lacks the vulkan driver for your GPU or your system doesn't have any vulkan-capable devices. If the device is known to be vulkan-capable try removing all GPU drivers and reinstalling/updating a driver for the device you want to test.
* `Runtime error: This device lacks support for DEVICE_LOCAL+HOST_COHERENT memory type.`<br>
Typical causes are software or hardware not supported by memtets_vulkan:
  * Emulator/translator usage: if the tested GPU name in the output mentions something like `Microsoft Direct3D12 (model name)` — the translator similar to Mesa Dozen “Vulkan-over-Direct3D12” is used, maybe installed as “OpenCL, OpenGL and Vulkan Compatibility Pack”. If several devices are listed on the memtest_vulkan startup, try selecting another driver variant.
  * Pre-2016 GPU: Some older GPUs are not supported due to driver limitations, like GTX780Ti on Windows even with the latest 472.xx driver
  * Old OS/driver: Windows 7 with 47x.xx driver has driver limitations even with post-2016 GPUs that are supported in newer environments.
* `Couldn't determine how much memory to test` on the integrated GPU <br>
If the integrated GPU is configured with fixed & quite low dedicated memory size - it may be shown in memtest_vulkan output only with 1GB VRAM: `1GB AMD Radeon(TM) Vega 3 Graphics` and then fail. The vulkan implementation for integrated GPUs allows using a bit less memory than reserved, and memtest_vulkan requires at least 1GB available memory to operate. Reconfigure integrated GPU to reserve at least 1.5GB of memory, see [issue #22](https://github.com/GpuZelenograd/memtest_vulkan/issues/22). The same message shows when too little system RAM is free for an integrated GPU: close other programs, or set the amount with `--size`.
* `the test couldn't run because of a runtime error` <br>
If the test fails to start and shows this message for a newer GPU - there is some incompatibility in vulkan installation. This may be caused by outdated driver or conflicts between several vulkan drivers installed. <br> <br>
For example on Linux the test can be run with a specific ICD the following way:<br>
`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./memtest_vulkan`<br>
[With Khronos vulkan loader `libvulkan.so` version below v1.3.207 use `VK_ICD_FILENAMES` instead of `VK_DRIVER_FILES`](https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.3.233/docs/LoaderInterfaceArchitecture.md#table-of-debug-environment-variables)<br>
Also try running with root/admin privileges - this is sometimes required on headless devices.

There are some reports that testing AMD GPUs sometimes gives unexpectedly low GPU load & video memory usage. The issue is still under investigation, but it is known that disabling/enabling "resizable BAR" in BIOS may help.

Also, some drivers don't allow single memory allocations or buffers bigger than ~4GB (`maxMemoryAllocationSize`/`maxBufferSize`), even on a GPU with a lot of memory. The tested memory is split into windows of at most 1GB, each with its own allocation and buffer kept within the limits the driver reports, so the whole memory is still tested.

If the test reports `WRITES NOT APPLIED`, it found a big contiguous block (1MB or more) of memory the test never managed to write to: it still holds the marker written before the test pattern, the data of the previous pass, or reads as all zeros. Faulty VRAM gives scattered errors instead, so this is most likely a GPU driver or memtest_vulkan problem and the results of that run are invalid. Try limiting the window size with `--window-mb 512` (or the `MEMTEST_VULKAN_MAX_WINDOW_MB` environment variable, rounded down to 128MB steps) or another driver version. If errors disappear with smaller windows, they were caused by the driver, not the memory. If they persist everywhere, the GPU itself may be faulty.

If nothing helps - enable verbose mode by renaming the executable to `memtest_vulkan_verbose` and running again. The test will output diagnostic information to stdout - please copy it to a new issue at https://github.com/SadCryingDuck/memtest_vulkan/issues.

<details><summary>

## <a id="development">New feature development</a>

</summary>
New ideas in the form of [suggestions via creating issues](https://github.com/SadCryingDuck/memtest_vulkan/issues) or pull requests are welcome. However, note that the tool is designed to be cross-platform, so an optimal way to add temperature and hardware monitoring is [relying on VK_KHR_performance_query extension](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPerformanceCounterUnitKHR.html), but unfortunately this extension isn't widely supported by 2022.

Note that portability is limited to 64-bit targets. Building versions for 32-bit targets was never tried, and all notices below assume 64-bit targets.

If you want to experiment with code modifications, there are two ways to do this:
 - the classic way: clone the repo, edit the code and build with a locally installed [Rust](https://rustup.rs) (`rust-toolchain.toml` selects the latest stable toolchain): `cargo build --release`. `./run-lint.sh` runs the same checks as CI: formatting, clippy and the unit tests.
      - Linux binaries for x86_64 and aarch64 are built with [cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild) against glibc 2.17, so they run on distributions back to 2014: `cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.17` (or `aarch64-unknown-linux-gnu.2.17`). This works from Windows, Linux and macOS.
      - Windows binaries link the C runtime statically (see `.cargo/config.toml`), so they don't need the Visual C++ Redistributable.
 - or the "fast way to make a small change relying on the github infrastructure". This can be useful if you are not familiar with building rust and don't plan to install it locally
      - fork this repository
      - enable workflows on the actions tab of the forked repository
      - edit&commit code changes (small changes are possible even via editing with a browser)
      - and github will build the binaries from your changes as artifacts on the actions tab in a few minutes!

### Releases

Releases are built and published by GitHub Actions (`.github/workflows/release.yml`) when a version tag is pushed:
1. set the new version in `Cargo.toml`, keeping the `-duck` suffix that marks this fork's builds, like `version = '0.10.1-duck'`, and commit it
2. tag that commit with the same version and push the tag: `git tag v0.10.1-duck` and `git push origin v0.10.1-duck`

The workflow checks that the tag matches the version, runs the checks and tests, builds Windows x86_64, Linux x86_64 and Linux aarch64 in parallel and publishes a release with these files, their SHA-256 checksums and the license texts:
 - `memtest_vulkan-v0.10.1-duck-windows-x86_64.exe`
 - `memtest_vulkan-v0.10.1-duck-linux-x86_64.tar.xz`
 - `memtest_vulkan-v0.10.1-duck-linux-aarch64.tar.xz`
 - `SHA256SUMS.txt`, `LICENSE.txt`, `LICENSE-zlib-original.txt`

It also signs a [build provenance attestation](https://docs.github.com/en/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds) for each binary, which records the workflow run and commit that built it. They're listed on the repository's [attestations page](https://github.com/SadCryingDuck/memtest_vulkan/attestations).

The Windows 7 builds of the original project need a special nightly toolchain and aren't made by this fork.

Since most of the time the GPUs are working fine, it may be hard to check the error handling behavior. The `MEMTEST_VULKAN_EMULATE_WRITE_BUG_ITERATION` environment variable was introduced to simplify this task. Set it to non-zero number generates 'fake' error during writing to the memory on the specified iteration that later would be found and reported during check stage.

Similarly, `MEMTEST_VULKAN_EMULATE_SKIPPED_WRITE_ITERATION` checks the `WRITES NOT APPLIED` detection: `0` skips the initial write of the last window, a positive number skips writing window 1 on the specified iteration. `MEMTEST_VULKAN_EMULATE_ALLOCATION_LIMIT` (like `6G`) makes bigger allocations fail as out of memory, to check the retry with less memory. `MEMTEST_VULKAN_EMULATE_FAILURE` makes a GPU pass fail with `device-lost`, `hang` or `panic`, optionally at a given pass like `hang:20` (the first passes are the initial fill, two per window), to check that failures end the test with a runtime error instead of being retried.
</details>

# Acknowledgements
The idea inspired by OpenCL-based cross-platform memory testing tool [memtestCL](https://github.com/ihaque/memtestCL).

The original implementation would not be possible without great vulkan bindings for rust provided by zlib-licensed [erupt library](https://gitlab.com/Friz64/erupt).
So, for licensing simplicity, the original memtest_vulkan is also licensed under the [zlib License](licenses/zlib-original.txt).
Since erupt is no longer maintained, this fork uses the [ash](https://github.com/ash-rs/ash) Vulkan bindings (MIT or Apache-2.0 licensed).

The `memtest_vulkan` itself was developed by [GpuZelenograd repair center](https://gpuzelenograd.github.io/README?memtest_vulkan)

# License
Copyright (C) 2026 SadCryingDuck

This project is licensed under the [GNU General Public License](LICENSE), version 3 or (at your option) any later version. This covers the whole project from this fork's first commit on.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

It is based on memtest_vulkan by GpuZelenograd, released under the zlib License, which allows this. The original zlib notice is kept unchanged in [licenses/zlib-original.txt](licenses/zlib-original.txt).
