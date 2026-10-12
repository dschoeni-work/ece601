# 滑鼠手寫辨識：RISC-V + 加速器

用滑鼠畫數字，由 Basys 3 FPGA 上的 RISC-V CPU 辨識，之後加上 systolic array 加速器量測加速比。

## 目錄

```
hw/rtl/sw_led.v                    LED 測試電路（確認燒錄流程）
hw/constraints/basys3_sw_led.xdc   Basys 3 腳位
model/train_quantize.py            訓練 MNIST MLP、INT8 量化、輸出 C 權重與測試資料
model/qmlp.py                      Python 整數推論，與 C 逐位元一致
firmware/mlp.h, mlp.c              C 整數推論，之後直接放到 RISC-V 上跑
firmware/test_mlp.c                在筆電上比對 C 與 Python 的結果
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
- [ ] 第 0 階段：滑鼠畫板與 MNIST 式前處理
- [ ] LED 測試上板（等 micro-USB 線）
- [ ] 第 1 階段：PicoRV32 + UART 上板辨識
