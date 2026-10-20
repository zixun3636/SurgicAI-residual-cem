# 擾動方向估計 + 方向條件式 CEM（2026/10）

## 要解決的問題
0918 的方向條件式 CEM 在 ±2° 內有 99–100% 成功率，但評估時是直接用「真實偏差 d 的正負」來選 θ_pos / θ_neg。實際部署時，系統事先並不知道偏差方向。

## 方法
旋轉動作偏差 d 在每個時間步都會加到動作上。Insert_env 會把實際執行的動作累加到 PSM 目標姿態，而 `observation[3:6]` 就是這個姿態（rad）。因此每一步：

    (rot_{t+1} − rot_t) / angle_step = clip(a_pre_t + d, −1, 1)

其中 a_pre_t 是控制器「想要」輸出的動作（Base + Residual + CEM，尚未加上偏差）。只要該軸沒有被限幅，

    d̂_t = (rot_{t+1} − rot_t) / angle_step − a_pre_t

就是 d 的一個估計樣本。流程如下：
1. 前 `warmup_steps` 步使用中性參數 θ = [1,1,1,0,0,0]（等同 Residual Only）。
2. 期間只收集沒有動作飽和（|執行動作| < 0.98）的樣本，把三軸樣本合併後取平均。
3. warm-up 結束時，依 sign(d̂) 鎖定使用 θ_pos 或 θ_neg。

整個過程不重新訓練任何網路，也不需要新的 CEM 搜尋，沿用 0918 的 θ_pos / θ_neg。

## 對照組（同 seed、同偏差條件）
| method | 說明 |
|---|---|
| oracle | 用真實 d 的正負選 θ（= 0918 的方向條件式 CEM，上界） |
| estimated | 用估計的 d̂ 選 θ（本次方法） |
| residual_only | 全程 θ = [1,1,1,0,0,0] |
| pos_only | 全程只用 θ_pos（單一組 CEM 參數） |

## 執行順序（在 WSL 的 ~/SurgicAI/RL 下）
| 步驟 | 指令 | 內容 | 大約 episode 數 |
|---|---|---|---|
| 0 | `./run_est_dir_smoke.sh` | 確認程式能跑，並估計每個 episode 花多少時間 | 24 |
| 1 | `nohup ./run_est_dir_main.sh > /dev/null 2>&1 &` | 主實驗：11 種偏差 × 4 種方法 × 100 ep | 4400 |
| 2 | `nohup ./run_est_dir_noise.sh > /dev/null 2>&1 &` | 姿態讀值雜訊 0.5°/1.0° × warm-up 5/10/20 | 1200 |
| 3 | `nohup ./run_est_dir_random3d.sh > /dev/null 2>&1 &` | （選做）隨機三軸偏差 | 600 |
| 4 | `./run_est_dir_plot.sh` | 出圖並印出表格 | – |

AMBF 只能同時跑一個模擬，所以步驟 1–3 請依序執行，不要同時跑。

- 結果：`evolution_results/estimated_direction/`（每跑完一個條件就存一次，中途中斷也不會遺失）
- log：`eval_logs_estimated_direction/`
- 圖：`figures/estimated_direction/`
- 用 `tail -f eval_logs_estimated_direction/main.log` 查看進度。

## 要回答的問題
1. estimated 的成功率能否接近 oracle？
2. 方向判斷正確率是多少？在 ±0.5° 這種小偏差下會不會出錯？
3. 和 pos_only 比較（單一組參數無法處理負向偏差）、和 residual_only 比較，各改善多少？
4. 加入姿態讀值雜訊後，需要多長的 warm-up 才能維持正確判斷？

## 已知限制（報告中要寫明）
- 模擬中的姿態讀值是「目標姿態」而非量測姿態，所以無雜訊時估計值幾乎是精確的。步驟 2 用人工加入的讀值雜訊模擬感測誤差，這是比較誠實的壓力測試。
- 偏差 d=0 時方向沒有正確答案，因此不計入方向正確率。
- 若某一軸在 warm-up 期間全部飽和，該軸就沒有樣本，估計只能依賴其他軸。
- 估計值假設三軸偏差相同（[d,d,d]），所以隨機三軸偏差只能得到一個合併後的正負號。

## 圖表若出現方框（缺少中文字型）
    sudo apt install fonts-noto-cjk && rm -rf ~/.cache/matplotlib
