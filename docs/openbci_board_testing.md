# Testing the OpenBCI boards (Cyton + Daisy)

A bench checklist for answering one question: **are the Cyton and Daisy boards
working, or is the problem in the wiring, adapter or cEEGrid?** Run it whenever a
channel looks dead, after any rewiring, and before resuming participant sessions.

The steps follow OpenBCI's own documentation and forum advice (links at the
bottom). What each fault looks like in our recordings, and how to repair the
cEEGrid side, is in [`recording_rig_faults.md`](recording_rig_faults.md).

## Why this is needed (status 2026-10-08)

| Problem | Channel(s) | Seen in |
|---|---|---|
| **L1 dead**, input disconnected | 1 | Intermittent since mid-2025 (fully dead in CTRL20, EXP46, EXP47, EXP52; partly in 7 more). **Dead in every session since the rewiring**: CTRL31, CTRL21 re-test, CTRL32 |
| **R7 dead**, input disconnected | 13 | Most sessions since 2025-02, including after the rewiring |
| **Right ear copies the left** | 9-16 | EXP47, CTRL27, CTRL31 |
| **Daisy sends only zeros** | 9-16 | EXP03 `-old` session (2025-04-17), EXP50, EXP52 |

A dead channel in these recordings sits at exactly **-187500 uV** for the whole
session. That is the lowest value the amplifier can report at gain x24 (the
range is +/-187.5 mV), and it's what an input with nothing connected to it does.
A bad electrode contact looks different: large and noisy, not a fixed number.

## Before you start

- **Battery power only.** Never connect the board to mains power while testing.
- **Never put a multimeter on electrodes attached to a person.** Do continuity
  checks with the cEEGrids off the participant and the board switched off.
- Open the OpenBCI GUI, connect to the Cyton + Daisy in **16-channel** mode, and
  turn on **Expert Mode** so you can send custom commands.
- Save your current hardware settings first (Hardware Settings -> Save). The
  tests below change channel settings, and you'll load them back at the end.
- For the square-wave tests: turn the GUI's **filters off** (band-pass and notch)
  and set the vertical scale to **Auto**.
- Write each result in the [results table](#results-table) as you go.

## Step 1: Is the Daisy detected?

Send the custom command **`C`** and read the reply:

| Reply | Meaning |
|---|---|
| `16$$$` or `daisy attached16$$$` | The Daisy is detected. Go on to step 2. |
| `no daisy to attach!8$$$` | The Daisy isn't detected. Reseat it on the Cyton's header pins, power-cycle, and try again before anything else. |

The board also looks for the Daisy every time it starts and defaults to 16
channels if it finds one. Our Daisy faults are intermittent (zeros in EXP50 and
EXP52, copies in EXP47, CTRL27 and CTRL31), so a loose Daisy-to-Cyton connection
is a suspect. Gently press the Daisy down on the Cyton and repeat `C`.

## Step 2: Board self-test (no electrodes involved)

These commands feed each channel a known signal from inside the amplifier chip,
bypassing every wire, the adapter and the cEEGrid. Send each one, wait a few
seconds, and read every channel's **uVrms** and **railed %** in the Time Series
view:

| Send | Test | A working Cyton or Daisy shows |
|---|---|---|
| `0` | Inputs tied to internal ground (noise floor) | 0.01% railed, **0.09-0.14 uVrms** |
| `-` | 1x square wave, slow | **1855-1865 uVrms**, 1.01% railed |
| `=` | 1x square wave, fast | same as `-` |
| `[` | 2x square wave, slow | **3680-3715 uVrms**, 2.01% railed |
| `]` | 2x square wave, fast | same as `[` |
| `p` | DC signal | 2.01% railed, 0.13-0.16 uVrms |

**How to read it:**
- **All 16 channels match the table:** both boards are fine. Any dead channel in
  real recordings is a wiring problem, so go to step 4.
- **Channel 1 or 13 fails here while the others pass:** that board's input is at
  fault. Email OpenBCI support (contact@openbci.com) with the readings.
- **Channels 9-16 all fail:** a Daisy problem. Redo step 1, then contact support.

## Step 3: Is the Daisy really independent?

Step 2 drives all channels at once, so it can't show whether channels 9-16 are
copies of channels 1-8. To check, put the test signal on **channel 1 only** and
leave the rest on normal input:

```
x1065010X
```

That's the SDK's channel-setting command: channel 1, on, gain 24, **test
signal**, bias excluded, SRB2 on, SRB1 off (your current settings apart from the
input type).

| What you see | Meaning |
|---|---|
| Square wave on channel 1 only | The Daisy reads its own inputs. Good. |
| Square wave on channel 1 **and channel 9** | Channels 9-16 are a copy of 1-8 inside the board or software. Contact support. |

Our duplicated recordings (EXP47, CTRL27, CTRL31) aren't bit-for-bit copies,
which points to the Daisy physically reading the left ear's electrodes rather
than a software copy. If this step passes, check how the right cEEGrid's cable
and jumper wires were plugged in for those sessions.

To check channel 13 on its own, use the Daisy letter for channel 13 (`Q W E R T Y U I`
are channels 9-16, so channel 13 is **T**): `xT065010X`.

**Restore afterwards:** send `d` (all channels back to OpenBCI defaults), then
**load your saved hardware settings** so bias and SRB2 return to your
configuration. `d` alone sets bias back to "include".

## Step 4: Find where the wiring breaks (if step 2 passed)

Work outward from the board toward the electrode, one channel at a time. The
path for each dead channel:

```
L1:  Cyton input 1 pin -> jumper wire -> adapter L1 pin -> left socket  -> L1 pad
R7:  Daisy input 5 pin -> jumper wire -> adapter R7 pin -> right socket -> R7 pad
```

**4a. Three-lead short (OpenBCI forum advice).** On the row of pins closest to
the board, connect the channel's pin, that board's **SRB2** pin and **BIAS**
together with a clip lead. The channel should read close to **0 uV**. Repeat the
short further along the path: at the adapter pin, then at the socket. The first
place where the channel goes back to -187500 uV is just past the break.

**4b. Continuity (board off, grids off the person).** With a multimeter in
continuity mode, check each hop above. Each hop should read a few ohms; the one
that reads open is the break. Also check the dead channel's pin against its
neighbours, SRB and BIAS: there should be **no** continuity there. The hop-by-hop
table for R7 is in
[`recording_rig_faults.md`, step 5](recording_rig_faults.md#step-5--continuity-test-to-localise-the-open).

**4c. Wiggle test.** With a gelled grid on a forearm and the GUI streaming,
gently flex the cEEGrid tail at the socket, the jumper wire and the socket body,
one at a time. If the channel jumps between dead and live at one spot, that's the
fault.

**4d. Swap test.** Swap the two cEEGrids between the ear sockets:

| Result | Fault is in |
|---|---|
| The dead channel moves with the grid | The cEEGrid (pad or flex trace) |
| It stays on the same channel number | The socket, adapter, jumper wire or board pin |

You can also move the dead channel's jumper to a working adapter pin (for
example L2's). If channel 1 comes alive, the board and jumper are fine.

**4e. Impedance check (on a person, after the above).** In the GUI's Cyton
Signal widget, use Impedance mode and check **one channel at a time**; OpenBCI
warns against checking several at once. With nothing attached, every channel
reads about **5000 kOhm**. On gelled skin, connected channels drop to roughly
10-50 kOhm (the range the cEEGrid reference design cites), while a broken channel
stays near the 5000 kOhm baseline.

## Step 5: Check the pins and settings

- **Pin rows:** electrodes go on the row of pins **closest to each board**, and
  channels 9-16 must be on the **Daisy**, not the Cyton. In one forum case, Daisy
  channels "railed" only because channels 9-16 were plugged into the Cyton.
- **Hardware settings** (our "2refs" file): all 16 channels on, gain x24, normal
  input, SRB2 connected, SRB1 disconnected. That's correct for the two-reference
  wiring.
- **Bias is excluded on every channel.** OpenBCI's default is "include" (and the
  saved file shows it used to be). Exclusion can't kill single channels, and 60 Hz
  noise in the new recordings is within the old range. Still, confirm it was
  switched off on purpose.

## Our current wiring (from 2026-09-17)

| | Default (repo design, until 2026-09-16) | Current ("New Bci Configuration") |
|---|---|---|
| Left ear reference (Cyton SRB2) | L6 | L6 |
| Right ear reference (Daisy SRB2) | L6, shared through the Y-splitter | **R6**, no Y-splitter |
| Bias / ground (Cyton BIAS) | R6 | **L5** |
| Channel 4 | L5 | **L3** |

Running the Cyton and Daisy with separate references isn't an OpenBCI-documented
setup. The standard is one reference shared through the Y-splitter, and that's
also what the cEEGrid repo describes. Nothing on the forum says separate
references won't work (each board references its own SRB2 pin), but nobody
reports using them. The forum's suggestions for a second reference are giving up
one channel to record the other earlobe, or re-referencing in software
afterwards.

## Results table

Copy this into your notes for each bench session:

| Date | Step | Channel(s) | Reading | Pass? | Notes |
|---|---|---|---|---|---|
| | 1 `C` | - | | | |
| | 2 `0` | all | | | |
| | 2 `-` | all | | | |
| | 3 `x1065010X` | 1 vs 9 | | | |
| | 4a short | 1 / 13 | | | |
| | 4b continuity | 1 / 13 | | | |
| | 4d swap | 1 / 13 | | | |

## Sources

OpenBCI documentation:
- [Cyton Board SDK](https://docs.openbci.com/Cyton/CytonSDK/): test-signal commands and expected values, channel-setting (`x`) format, Daisy channel letters, `C`/`c` replies, impedance (`z`) command
- [OpenBCI GUI Widget Guide](https://docs.openbci.com/Software/OpenBCISoftware/GUIWidgets/): hardware-settings input types, impedance mode, railed indicators
- [Daisy getting-started guide](https://docs.openbci.com/GettingStarted/Boards/DaisyGS): standard Y-splitter wiring

OpenBCI forum:
- [openbci-ceegrids Cyton+Daisy GUI](https://openbci.com/forum/index.php?p=%2Fdiscussion%2F3887%2Fopenbci-ceegrids-cythondaisy-board-gui): the same cEEGrid adapter; dead Daisy channels; intermittent Daisy connection
- [Channels 9-16 railing, Y-cable issue?](https://openbci.com/forum/index.php?p=%2Fdiscussion%2F1510%2Fchannels-9-16-railing-y-cable-issue): three-lead short test; channels 9-16 on the wrong board
- [Cyton picking up noise only on one channel?](https://openbci.com/forum/index.php?p=/discussion/3737/cyton-picking-up-noise-only-on-one-channel): floating inputs; when to contact support
- [Test Signal](https://openbci.com/forum/index.php?p=/discussion/2332/test-signal): a working single-channel test-signal command
- [Averaged or single reference electrode?](https://openbci.com/forum/index.php?p=/discussion/1195/averaged-or-single-reference-electrode): two-reference alternatives
- [The beginner's wiring question (Cyton + Daisy)](https://openbci.com/forum/index.php?p=/discussion/1016/the-beginners-wiring-question-cyton-daisy): SRB2 / Y-cable / BIAS roles
- [All channels with the same signal](https://openbci.com/forum/index.php?p=%2Fdiscussion%2F3816%2Fall-channels-with-the-same-signal): identical channels from a shared reference problem

cEEGrid adapter:
- [MKnierim/openbci-ceegrids](https://github.com/MKnierim/openbci-ceegrids): the reference design our default wiring follows
