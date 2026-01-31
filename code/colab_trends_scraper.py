# 此脚本设计用于 Google Colab 环境运行
# ==========================================
# 使用说明：
# 1. 点击播放按钮运行代码
# 2. 点击 "选择文件" 上传包含 Name, Start Date, End Date 的 CSV 文件
# 3. 脚本会自动循环爬取，直到所有数据都获取成功
#    - 成功获取数值 -> 保存
#    - Google返回无数据 -> 记为 0 (不再重试)
#    - 报错/被墙/超时 -> 保持为空 (NaN)，下一轮自动重试
# 4. 最终会自动下载 'Final_DWTS_Popularity.csv'
# ==========================================

import pandas as pd
from pytrends.request import TrendReq
from google.colab import files
import time
import random
import io
import os

# ================= 配置区 =================
OUTPUT_FILENAME = 'Final_DWTS_Popularity.csv'       # 最终结果文件名
PARTIAL_FILENAME = 'DWTS_Progress.csv'              # 中间进度文件名 (断点续传用)
RETRY_ZEROS = False                                 # 【重要】如果你的文件中 "没爬到的" 显示为 0，请改为 True。
                                                    # 如果 False，则只重试空值 (NaN)，跳过 0 值。
# =========================================

# 1. 安装库 (如尚未安装)
try:
    import pytrends
except ImportError:
    print("正在安装 pytrends...")
    !pip install pytrends
    from pytrends.request import TrendReq

# 2. 上传文件
print(f"请点击下方按钮上传 CSV 文件 (支持原始文件或 '{PARTIAL_FILENAME}' 断点续传)...")
uploaded = files.upload()
if not uploaded:
    print("未上传文件，程序结束。")
else:
    filename = list(uploaded.keys())[0]
    print(f"已读取: {filename}")
    
    # 3. 读取数据
    df = pd.read_csv(io.BytesIO(uploaded[filename]))

    # 初始化列
    if 'Average_Popularity_Score' not in df.columns:
        df['Average_Popularity_Score'] = None
    
    # 如果用户指出失败的也是0，且启用了重试0的选项，这里将0转回NaN以便重跑
    if RETRY_ZEROS:
         # 将 0 视为未完成
        print("⚠️ 注意：RETRY_ZEROS = True。所有值为 0 的项都将被重新爬取！")
        # 仅将严格等于 0.0 的转为 NaN，避免误伤
        mask = (df['Average_Popularity_Score'] == 0) | (df['Average_Popularity_Score'] == 0.0)
        df.loc[mask, 'Average_Popularity_Score'] = None

    print(f"数据加载完成，共 {len(df)} 行。")

    # 4. 核心处理函数
    def process_batch(dataframe):
        # 初始化 pytrends (添加更多重试参数)
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), retries=2, backoff_factor=1)
        
        # 待处理列表：所有值为空的行
        pending_mask = dataframe['Average_Popularity_Score'].isna()
        pending_indices = dataframe[pending_mask].index.tolist()
        
        if not pending_indices:
            return False # 任务全部完成
        
        print(f"\n--- 本轮剩余 {len(pending_indices)} 条数据待爬取 ---")
        
        progress = False
        
        for i, idx in enumerate(pending_indices):
            row = dataframe.loc[idx]
            name = row['Name']
            # 确保时间格式正确
            time_range = f"{row['Start Date']} {row['End Date']}"
            
            print(f"[{i+1}/{len(pending_indices)}] 查询: {name} ...", end=" ")
            
            try:
                # 随机延迟开头，模拟人类操作
                time.sleep(random.uniform(2, 5))
                
                # 构建请求
                pytrends.build_payload([name], cat=0, timeframe=time_range, geo='US', gprop='')
                data = pytrends.interest_over_time()
                
                if not data.empty:
                    # 检查是否包含该列
                    if name in data.columns:
                        avg_score = data[name].mean()
                        print(f"✅ 获取: {avg_score:.2f}")
                        dataframe.at[idx, 'Average_Popularity_Score'] = avg_score
                    else:
                        # 极少数情况有数据但列名不匹配
                        print(f"⚠️ 数据格式异常 (重试)")
                        continue 
                else:
                    print(f"⚪ 无数据 (记为0)")
                    dataframe.at[idx, 'Average_Popularity_Score'] = 0.0
                
                # 只要没抛错，就算有进度
                progress = True
                
                # 每 5 条保存一次中间结果
                if (i + 1) % 5 == 0:
                    dataframe.to_csv(PARTIAL_FILENAME, index=False)
                
            except Exception as e:
                err = str(e)
                if "429" in err:
                    print(f"❌ 429 限流 (暂停 60s)")
                    time.sleep(60)
                else:
                    print(f"❌ 错误: {err}")
                    time.sleep(5)
                # 遇到错误保持 NaN，下轮继续
        
        return True # 本轮循环结束

    # 5. 主循环
    round_count = 1
    while True:
        try:
            # 检查剩余空缺
            nan_count = df['Average_Popularity_Score'].isna().sum()
            if nan_count == 0:
                print("\n🎉 全部完成！所有数据已获取。")
                break
            
            print(f"\n====== 第 {round_count} 轮迭代 ======")
            has_pending = process_batch(df)
            
            if not has_pending:
                break
            
            # 保存本轮结果
            df.to_csv(PARTIAL_FILENAME, index=False)
            print(f"本轮进度已保存到 {PARTIAL_FILENAME}")
            
            # 增加轮次
            round_count += 1
            
            # 轮次间额外休息
            print("轮次间休息 10 秒...")
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n🛑 用户手动停止")
            break
        except Exception as e:
            print(f"\n❌ 主循环异常: {e}")
            break

    # 6. 下载结果
    print(f"\n正在下载最终结果: {OUTPUT_FILENAME}")
    df.to_csv(OUTPUT_FILENAME, index=False)
    try:
        files.download(OUTPUT_FILENAME)
    except Exception as e:
        print("下载失败，请手动在左侧文件栏下载。")
