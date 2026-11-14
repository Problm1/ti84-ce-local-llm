# TI-84 Plus CE Local Micro-LLM (Offline Transformer Chatbot)

A genuine, 100% offline **Llama-architecture Transformer neural network** running natively on an unmodified **TI-84 Plus CE** graphing calculator (48 MHz Zilog eZ80 CPU, 154 KB RAM, 3.0 MB Flash ROM)—with **zero Wi-Fi mods, external microcontrollers, or PC tethering**.

Because a 300K parameter model is way too small to reliably solve equations, it was specifically trained to hold basic small talk and **cheerfully refuse math or logic questions** (on a graphing calculator, naturally).

## How It Works
Running a generative language model on a stock TI-84 Plus CE requires solving a severe memory wall:
* **8-Bit Quantized Micro-Transformer:** Built on Andrej Karpathy's `llama2.c` v2 format (260K parameters, 4 layers, 64 dimensions, 4 attention heads, 512-token BPE vocabulary) running completely offline on the eZ80 processor.
* **6-AppVar Flash Streaming:** Because TI-OS caps individual AppVars at 64 KB, the 316.5 KB of quantized model weights are sliced into six chunks (`MODEL0.8xv`–`MODEL4.8xv` at 60.1 KB each, plus `MODEL5.8xv` at 16 KB) stored in Archive Flash and streamed on the fly via `<fileioc.h>`.
* **Zero-Heap Shadow RAM Mapping:** To avoid exhausting the calculator's 154 KB user RAM, the vocabulary table and KV-cache buffers are mapped directly into off-screen TI-OS shadow memory (`pixelShadow` at `0xD031F6` and `saveSScreen` at `0xD0EA1F`).

## Installation (Run on Your Own TI-84 Plus CE)
1. **Jailbreak (OS 5.5+):** Ensure your calculator can run native C/Assembly programs using **arTIfiCE** and **Cesium**.
2. **Transfer Files:** Using TI Connect CE or TiLP, send **`CHATBOT.8xp`**, **`TOKEN.8xv`**, and all **`MODEL0.8xv` – `MODEL5.8xv`** files to your calculator.
3. **Archive AppVars:** Press `2nd` -> `+` (`MEM`) -> `2: Mem Mgmt/Del...` -> `8: AppVars...` and archive (`*`) **`TOKEN`** and every **`MODEL`** file (`MODEL0` through `MODEL5`) so they sit in Flash ROM instead of RAM.
4. **Chat:** Launch **`CHATBOT`** from Cesium. Type a short lowercase prompt (press `0` for spaces), hit `enter`, and wait ~1-3 Minutes as the 48 MHz CPU crunches through the transformer layers.

## Current Limitations (v0.1)
* **Speed:** Inference takes roughly 1-3 Minutes per response on the 48 MHz CPU.
* **Coherence:** Generates text and numbers for basic small talk, though clean sentence structure is still rough. Pull requests to optimize Flash pointer caching or speed up the dot-product loop are welcome!

## Building from Source
1. Install the **[CE C/C++ Toolchain](https://github.com)**.
2. Place `main.c` inside a `src/` folder alongside `Makefile`.
3. Run `make` to compile `bin/CHATBOT.8xp`.

## Credits & Acknowledgments
* **[llama2.c](https://github.com)** by Andrej Karpathy for the core Llama transformer architecture, BPE tokenizer format, and 8-bit quantization exporter.
* **[CE Programming Toolchain](https://github.com)** for the Zilog eZ80 C compiler and TI-84 Plus CE libraries (`<fileioc.h>`, `<ti/screen.h>`, `<ti/getcsc.h>`).
* **Development & AI Collaboration:** Created by a student working on a budget HP Stream 14 laptop, with AI coding assistance (Gemini) used to help design the 6-AppVar streaming architecture, map TI-OS shadow memory buffers, and write/debug the C implementation.

