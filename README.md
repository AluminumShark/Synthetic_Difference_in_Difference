# SyntheticDiffInDiff | 合成雙重差分法

[English](#english) | [繁體中文](#繁體中文)

---

## English

Hey there, fellow researcher! 👋 

Are you tired of trying to figure out if that policy change actually worked? Struggling with messy panel data that doesn't quite fit into traditional analysis frameworks? Well, you've come to the right place!

This is a Python implementation of **Synthetic Difference-in-Differences (SDID)** - think of it as the Swiss Army knife of causal inference. It's like having the best parts of synthetic control methods and difference-in-differences analysis rolled into one awesome tool.

### 🤔 What's the Big Deal?

You know how sometimes you have treatment and control groups, but they're just... different? And you're not sure if the changes you see are because of your intervention or just because these groups were always different? That's where SDID comes in clutch.

SDID is smart - it figures out how to weight your control units and time periods to create the most realistic counterfactual. It's like having a crystal ball that shows you "what would have happened if we didn't do anything."

Perfect for:
- Policy evaluation (did that new regulation actually help?)
- Business analytics (was our marketing campaign worth it?)
- Academic research (proving your theory with solid data)
- Any situation where you need to isolate causal effects

### ✨ What Makes This Tool Special?

Look, I've used a lot of econometric tools, and most of them make you feel like you need a PhD just to get started. Not this one:

- **Smart Weight Calculation**: The algorithm figures out the optimal weights automatically - no guesswork!
- **Event Study Magic**: Want to see how effects evolved over time? We've got you covered
- **Confidence Intervals**: Because we all know point estimates without uncertainty bounds are basically useless
- **Pretty Plots**: Publication-ready visualizations that'll make your advisor smile
- **Actually Readable Code**: Type hints, logging, and documentation that doesn't require a decoder ring
- **Parallel Processing**: Because life's too short to wait for bootstrap samples

### 📦 Getting Started (The Easy Way)

First things first - let's get you set up:

```bash
# Install the good stuff
pip install numpy pandas cvxpy statsmodels joblib matplotlib scipy

# Grab our code
git clone https://github.com/yourusername/Synthetic_Difference_in_Difference.git
cd Synthetic_Difference_in_Difference
```

### 🚀 Your First Analysis (5 Minutes or Less)

Here's how easy it is to get meaningful results:

```python
import pandas as pd
from SDID import SyntheticDiffInDiff

# Load your data (we'll assume you have it ready)
data = pd.read_csv('your_data.csv')

# Set up the analysis - just tell us which column is which
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col='outcome',      # What you're measuring
    times_col='time',           # When stuff happened
    units_col='unit',           # Who we're looking at
    treat_col='treated',        # Who got the treatment
    post_col='post'            # When treatment happened
)

# Get your answer!
treatment_effect = sdid.run_analysis()
print(f"Boom! Your treatment effect is: {treatment_effect:.4f}")

# Want to know how confident you should be?
sdid.estimate_se(bootstrap_rounds=200, n_jobs=4)  # Use 4 cores because why not?
print(f"Standard error: {sdid.standard_error:.4f}")
```

That's it! You just did causal inference like a boss. 🎉

### 📊 Going Deeper (For the Curious)

#### Want to see how effects changed over time?

```python
# Event study analysis - see the story unfold
post_periods = data[data['post']]['time'].unique()
effects = sdid.run_event_study(post_periods)

# Make it look professional
fig = sdid.make_figure(
    times=post_periods,
    bootstrap_rounds=100,
    n_jobs=4,  # Parallel processing FTW
    confidence_level=0.95,
    figure_size=(14, 8)  # Big enough to see the details
)
fig.savefig('my_awesome_results.png', dpi=300, bbox_inches='tight')
```

#### Curious about which units/times matter most?

```python
# Peek under the hood
weights_summary = sdid.get_weights_summary()

print("These control units carried the most weight:")
print(weights_summary['unit_weights'].head())

print("\nThese time periods were most important:")
print(weights_summary['time_weights'].head())
```

#### Want the full diagnostic report?

```python
# Get all the juicy details
diagnostics = sdid.get_diagnostics()

print(f"Treatment Effect: {diagnostics['treatment_effect']:.4f}")
print(f"How confident are we? SE = {diagnostics['standard_error']:.4f}")
print(f"Effective control units: {diagnostics['effective_control_units']}")
print(f"Effective time periods: {diagnostics['effective_time_periods']}")
```

### 📋 What Your Data Should Look Like

Don't overthink this - just make sure your data is in **long format** (one row per unit-time combination):

| Column | What It Is | Example |
|--------|------------|---------|
| Outcome | The thing you care about | Sales, GDP, Test Scores |
| Time | When it happened | 2018, Q1, Jan |
| Unit | Who/what you're studying | States, Companies, Schools |
| Treatment | Who got treated | 1 = treated, 0 = control |
| Post | When treatment started | 1 = after treatment, 0 = before |

Here's what good data looks like:
```
unit | time | outcome | treated | post
-----|------|---------|---------|-----
CA   | 2018 | 100.5   | 0       | 0    (California, pre-treatment, control)
CA   | 2019 | 102.3   | 0       | 0    
CA   | 2020 | 105.1   | 0       | 1    (Now it's post-treatment)
TX   | 2018 | 98.2    | 1       | 0    (Texas got the treatment)
TX   | 2019 | 99.8    | 1       | 0    
TX   | 2020 | 108.7   | 1       | 1    (Post-treatment for treated unit)
```

### 🔬 The Science Behind the Magic

If you're curious about what's happening under the hood (and you should be!):

1. **Step 1 - Unit Weights**: "Which control units look most like our treated units before treatment?"
2. **Step 2 - Time Weights**: "Which time periods give us the most balanced comparison?"
3. **Step 3 - The Big Calculation**: Combine everything into a weighted difference-in-differences

The math includes smart regularization to prevent overfitting - because we've all seen models that work perfectly on training data and terribly everywhere else.

### 📖 Give Credit Where It's Due

This implementation is based on the brilliant work by Arkhangelsky et al. If you use this in your research, please cite the original paper:

```bibtex
@article{arkhangelsky2021synthetic,
  title={Synthetic difference-in-differences},
  author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A and Imbens, Guido W and Wager, Stefan},
  journal={American Economic Review},
  volume={111},
  number={12},
  pages={4088--4118},
  year={2021}
}
```

### 🤝 Found a Bug? Have an Idea?

We're human (mostly), so there are probably bugs. If you find one, or if you have ideas for making this better, please let us know! Open an issue, submit a PR, or just send us a message. We're friendly, I promise.

### ⚠️ A Quick Reality Check

SDID is powerful, but it's not magic. Like any causal inference method, it relies on assumptions:
- Your control units provide a good counterfactual
- Treatment assignment isn't based on anticipation of future outcomes
- No spillover effects between units

Always think critically about whether these assumptions hold in your context!

---

## 繁體中文

嘿，研究夥伴們！👋

還在為政策效果評估而煩惱嗎？面對凌亂的面板數據不知道該用什麼分析方法？你來對地方了！

這是一個**合成雙重差分法（SDID）**的Python實作工具 - 把它想像成因果推論的瑞士軍刀。就像把合成控制法和雙重差分法的精華部分完美結合在一起。

### 🤔 為什麼要用這個？

你知道有時候處理組和控制組就是...不太一樣嗎？你不確定看到的變化是因為你的干預措施，還是因為這些組本來就不同？這就是SDID發揮作用的時候了。

SDID很聰明 - 它會想辦法為你的控制單位和時間期間分配權重，創造出最真實的反事實情境。就像有一個水晶球告訴你「如果我們什麼都沒做會發生什麼」。

特別適合：
- 政策評估（新規定真的有幫助嗎？）
- 商業分析（我們的行銷活動值得嗎？）
- 學術研究（用扎實的數據證明你的理論）
- 任何需要分離因果效應的情況

### ✨ 這個工具的特別之處

老實說，我用過很多計量經濟學工具，大部分都讓人覺得需要博士學位才能開始使用。但這個不一樣：

- **智慧權重計算**：演算法自動找出最佳權重 - 不用猜測！
- **事件研究分析**：想看效果如何隨時間演變？我們幫你搞定
- **信賴區間**：因為沒有不確定性邊界的點估計基本上沒用
- **美麗圖表**：可以發表的視覺化圖表，讓你的指導教授滿意
- **真正可讀的程式碼**：型別提示、日誌記錄和不需要解碼器的文件
- **平行處理**：因為人生苦短，不該浪費時間等bootstrap樣本

### 📦 開始使用（簡單方式）

首先，讓我們設定環境：

```bash
# 安裝必要套件
pip install numpy pandas cvxpy statsmodels joblib matplotlib scipy

# 下載我們的程式碼
git clone https://github.com/yourusername/Synthetic_Difference_in_Difference.git
cd Synthetic_Difference_in_Difference
```

### 🚀 你的第一次分析（5分鐘內完成）

看看獲得有意義的結果有多簡單：

```python
import pandas as pd
from SDID import SyntheticDiffInDiff

# 載入你的數據（假設你已經準備好了）
data = pd.read_csv('your_data.csv')

# 設定分析 - 只要告訴我們哪一欄是什麼
sdid = SyntheticDiffInDiff(
    data=data,
    outcome_col='outcome',      # 你要測量的東西
    times_col='time',           # 什麼時候發生的
    units_col='unit',           # 我們在看誰
    treat_col='treated',        # 誰接受了處理
    post_col='post'            # 處理什麼時候發生的
)

# 得到你的答案！
treatment_effect = sdid.run_analysis()
print(f"太棒了！你的處理效果是：{treatment_effect:.4f}")

# 想知道你應該多有信心？
sdid.estimate_se(bootstrap_rounds=200, n_jobs=4)  # 用4個核心，為什麼不呢？
print(f"標準誤：{sdid.standard_error:.4f}")
```

就是這樣！你剛剛像專家一樣做了因果推論。🎉

### 📊 深入探索（給好奇的你）

#### 想看效果如何隨時間變化？

```python
# 事件研究分析 - 看故事如何展開
post_periods = data[data['post']]['time'].unique()
effects = sdid.run_event_study(post_periods)

# 讓它看起來專業
fig = sdid.make_figure(
    times=post_periods,
    bootstrap_rounds=100,
    n_jobs=4,  # 平行處理萬歲
    confidence_level=0.95,
    figure_size=(14, 8)  # 大到能看清楚細節
)
fig.savefig('我的超棒結果.png', dpi=300, bbox_inches='tight')
```

#### 好奇哪些單位/時間點最重要？

```python
# 窺探引擎蓋下面
weights_summary = sdid.get_weights_summary()

print("這些控制單位權重最高：")
print(weights_summary['unit_weights'].head())

print("\n這些時間期間最重要：")
print(weights_summary['time_weights'].head())
```

### 📋 你的數據應該長什麼樣子

別想太複雜 - 只要確保你的數據是**長格式**（每個單位-時間組合一行）：

| 欄位 | 是什麼 | 例子 |
|------|--------|------|
| Outcome | 你關心的東西 | 銷售額、GDP、考試成績 |
| Time | 什麼時候發生的 | 2018、第一季、一月 |
| Unit | 你在研究誰/什麼 | 州、公司、學校 |
| Treatment | 誰被處理了 | 1 = 處理組，0 = 控制組 |
| Post | 處理什麼時候開始 | 1 = 處理後，0 = 處理前 |

### 🔬 魔法背後的科學

如果你好奇引擎蓋下發生了什麼（你應該好奇！）：

1. **步驟1 - 單位權重**：「哪些控制單位在處理前最像我們的處理單位？」
2. **步驟2 - 時間權重**：「哪些時間期間給我們最平衡的比較？」
3. **步驟3 - 大計算**：將一切結合成加權雙重差分

數學包含智慧正規化以防止過擬合 - 因為我們都見過在訓練數據上完美但在其他地方糟糕的模型。

### 📖 給予應有的信用

這個實作基於Arkhangelsky等人的出色工作。如果你在研究中使用這個，請引用原始論文：

```bibtex
@article{arkhangelsky2021synthetic,
  title={Synthetic difference-in-differences},
  author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A and Imbens, Guido W and Wager, Stefan},
  journal={American Economic Review},
  volume={111},
  number={12},
  pages={4088--4118},
  year={2021}
}
```

### 🤝 發現bug了？有想法？

我們是人類（大部分），所以可能有bug。如果你發現了，或者有讓這個工具更好的想法，請告訴我們！開個issue、提交PR，或者直接給我們留言。我們很友善的，我保證。

### ⚠️ 現實檢查

SDID很強大，但不是魔法。像任何因果推論方法一樣，它依賴假設：
- 你的控制單位提供了良好的反事實
- 處理分配不是基於對未來結果的預期
- 單位之間沒有溢出效應

在你的情境中，總是批判性地思考這些假設是否成立！

---

*建造時充滿愛心 ❤️，由開源社群貢獻* 