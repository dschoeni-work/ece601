# 滑鼠手寫辨識：RISC-V + 加速器

用滑鼠畫數字，由 Basys 3 FPGA 上的 RISC-V CPU 辨識，之後加上 systolic array 加速器量測加速比。

## 目錄

```
hw/rtl/sw_led.v                    LED 測試電路（確認燒錄流程）
hw/constraints/basys3_sw_led.xdc   LED 測試的 Basys 3 腳位
hw/rtl/picorv32.v                  PicoRV32 RISC-V CPU（YosysHQ，ISC 授權，未修改）
hw/rtl/soc.v                       SoC：CPU + 128 KB RAM + UART + LED/七段顯示器暫存器
hw/rtl/uart.v                      UART 收發（8N1）
hw/rtl/basys3_top.v                Basys 3 頂層：時脈、重置按鈕、七段顯示器掃描
hw/constraints/basys3_soc.xdc      SoC 的 Basys 3 腳位
firmware/start.S, link.ld          開機程式與記憶體配置
firmware/main.c                    韌體：從 UART 收圖、推論、回傳結果與 cycle 數
firmware/Makefile                  編譯韌體成 build/firmware.hex（make）與筆電測試（make test）
sim/sim_main.cpp, sim/Makefile     Verilator 模擬：送 MNIST 圖進 SoC，比對 Python、量 cycle 數
model/train_quantize.py            訓練 MNIST MLP、INT8 量化、輸出 C 權重與測試資料
model/qmlp.py                      Python 整數推論，與 C 逐位元一致
firmware/mlp.h, mlp.c              C 整數推論，之後直接放到 RISC-V 上跑
firmware/test_mlp.c                在筆電上比對 C 與 Python 的結果
host/draw_digit.py                 滑鼠畫板：畫數字、看模型輸入與即時辨識結果
host/preprocess.py                 把滑鼠筆跡轉成 MNIST 格式的 28×28 圖
host/test_preprocess.py            檢查前處理（不需要開視窗）
host/eval_my_digits.py             計算模型在自己手寫數字上的準確率
host/my_digits/                    畫板存下的手寫數字（筆跡 JSON）
```

## 第 0 階段：執行方式

**1. 安裝 NumPy**（唯一需要的套件）

```
pip install numpy
```

Windows 若還沒有 Python，從 python.org 下載安裝，安裝畫面記得勾選「Add python.exe to PATH」。

**2. 訓練與量化**

```
cd model
python train_quantize.py
```

會自動下載 MNIST（約 11 MB），訓練約 30 秒。參考結果：float 97.72%、INT8 97.73%。
輸出 `firmware/mlp_weights.h`、`firmware/test_data/`，以及給之後滑鼠畫板用的 `model/quant_model.npz`。

**3. 編譯 C 並比對**

```
cd ../firmware
gcc -O2 -o test_mlp test_mlp.c mlp.c
./test_mlp
```

成功時最後一行是 `PASS: C matches Python bit for bit`，代表 10,000 張測試圖的每一個 logit 都和 Python 一致。

需要 C 編譯器：Mac 執行 `xcode-select --install` 即可；Windows 等 WSL 裝好後在 Ubuntu 裡執行同樣的指令。

## 滑鼠畫板

需要 Tkinter（Python 內建的視窗套件）。Windows 的 Python 已內建；Mac 用 Homebrew 的 Python 時要另外裝，版本號換成你的 Python 版本：

```
brew install python-tk@3.14
```

執行（需先跑過 `model/train_quantize.py`）：

```
cd host
python draw_digit.py
```

| 操作 | 功能 |
| --- | --- |
| 按住左鍵拖曳 | 畫 |
| Space / Delete | 清除 |
| Backspace / u | 復原上一筆 |
| 0–9 | 用該數字當標籤存到 `host/my_digits/`，然後清除 |
| x | 刪除剛剛存的那一張（按錯數字時用） |

右邊會顯示模型實際看到的 28×28 圖、辨識結果和每個數字的信心度。辨識用的是和 C 韌體相同的整數模型，所以這裡的答案就是 FPGA 上會得到的答案。

**前處理**（`preprocess.py`）照 MNIST 的做法：把數字等比縮放到最長邊 20 px（灰階邊緣）、放進 28×28，再依重心移到 (14, 14)。筆畫粗細依畫的大小調整，畫大畫小結果相同。

**測自己的手寫**：每個數字畫 10 張左右並用數字鍵存下，再執行 `python eval_my_digits.py`。存的是原始筆跡，之後改了前處理或重新訓練，再跑一次就能比較。

`python test_preprocess.py` 會檢查前處理：把 MNIST 測試圖隨機放大、移位後再處理，準確率應該不變（參考：97.50% → 97.65%）。

## 第 1 階段：RISC-V SoC 模擬

用 Verilator 模擬整顆 SoC：PicoRV32 執行 `mlp.c`，測試程式扮演筆電，透過 UART 把 MNIST 測試圖送進去，收回結果後跟 Python 逐位元比對，並回報每張圖的 cycle 數。

**安裝工具**（Mac）：

```
brew install riscv64-elf-gcc verilator
```

Ubuntu / WSL：`sudo apt install gcc-riscv64-unknown-elf verilator`。Makefile 會自動找到對應的編譯器名稱。

**執行**（需先跑過 `model/train_quantize.py`）：

```
make -C firmware
make -C sim run
make -C sim run N=200     # 測更多張
```

參考結果（GCC 13，Verilator 5.020）：

| 項目 | 結果 |
| --- | --- |
| Logits 與 Python 一致 | 100 / 100 張 |
| 每張 cycle 數 | 約 2,188,000 |
| 100 MHz 時每張時間 | 約 21.9 ms |
| 韌體大小（程式 + 權重） | 52 KB |

約 5 萬次乘加、每次約 43 cycle：內層迴圈每次乘加要 8 個指令，PicoRV32 平均每個指令 4–5 cycle。換不同版本的編譯器，cycle 數可能差幾 %，但 logits 一定要完全一致。

**SoC 設計**

- CPU：PicoRV32，RV32IM，開啟硬體乘法器（DSP）和 barrel shifter，作為公平的純 CPU 基準。
- 記憶體：128 KB 區塊 RAM，程式、權重和 stack 都在裡面，開機時從 `firmware.hex` 載入。
- UART：115200 baud。模擬時把每個 bit 縮短成 16 個 cycle 以加快速度，也用真實的 868 cycle 驗證過。
- 計時：韌體用 `rdcycle` 量 `mlp_infer()` 前後的 cycle 差，只算推論本身，不含 UART 傳輸。

**UART 協定**（數字皆為 little-endian）

| 方向 | 內容 |
| --- | --- |
| 筆電 → 板子 | `'P'`（ping） |
| 板子 → 筆電 | `'K'` |
| 筆電 → 板子 | `'I'` + 784 bytes（28×28 圖） |
| 板子 → 筆電 | `'R'`、數字（1 byte）、cycle 數（uint32）、10 個 logits（int32），共 46 bytes |

板子收到圖後也會把結果顯示在最右邊的七段顯示器，並點亮對應編號的 LED；LED 15 亮代表 CPU 出錯（trap）。

## 量化方式

| 步驟 | 運算 | 資料型別 |
| --- | --- | --- |
| 輸入 | `x = pixel >> 1`（0–127） | uint8 |
| 第 1 層 | `acc = Σ x·W1 + B1` | INT8 × INT8 → INT32 |
| 重新量化 + ReLU | `h = clamp((acc·M1 + 2^(S−1)) >> S, 0, 127)` | INT64 乘法 → INT8 |
| 第 2 層 | `logit = Σ h·W2 + B2` | INT8 × INT8 → INT32 |
| 輸出 | `argmax(logit)`，同分取第一個 | — |

- 全程沒有浮點運算，`mlp.c` 也不用 libc，可以直接編譯成 RV32I。
- 編成 RV32I 約 52 KB（程式加權重），Basys 3 的 BRAM 約 225 KB，放得下。
- CPU 沒有乘法器（純 RV32I）時，連結時加 `-lgcc` 使用軟體乘法；若 PicoRV32 開啟乘法器，改用 `-march=rv32im`。

## 進度

- [x] 第 0 階段：訓練、INT8 量化、C 推論與 Python 逐位元一致
- [x] 第 0 階段：滑鼠畫板與 MNIST 式前處理
- [ ] 第 0 階段：收集自己的手寫數字，測準確率
- [ ] LED 測試上板（等 micro-USB 線）
- [x] 第 1 階段：PicoRV32 SoC 模擬，與 Python 逐位元一致，純 CPU 約 2.19M cycle／張
- [ ] 第 1 階段：Vivado 合成上板，畫板透過 UART 送圖給板子辨識
