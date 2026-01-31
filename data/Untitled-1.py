import pandas as pd
from pytrends.request import TrendReq
import time
import random

# 读取数据
df = pd.read_csv('Dancing_with_the_Stars_Trends_Setup.csv') # 确保文件名一致

# 连接 Google Trends
pytrends = TrendReq(hl='en-US', tz=360)

print(f"总共需要查询 {len(df)} 条数据，预计耗时约 {len(df)*3/60:.1f} 分钟...")

results = []

for index, row in df.iterrows():
    name = row['Name']
    # 组合成 Google Trends 需要的时间格式: "YYYY-MM-DD YYYY-MM-DD"
    time_range = f"{row['Start Date']} {row['End Date']}"
    
    try:
        # 构建查询
        pytrends.build_payload([name], cat=0, timeframe=time_range, geo='US', gprop='')
        
        # 获取数据
        data = pytrends.interest_over_time()
        
        if not data.empty:
            # 计算该时间段的平均热度
            avg_score = data[name].mean()
            print(f"[{index+1}/{len(df)}] {name} (第{row['Season']}季): {avg_score:.2f}")
            results.append(avg_score)
        else:
            print(f"[{index+1}/{len(df)}] {name}: 无数据")
            results.append(0)
            
        # 休息 2-4 秒，防止被封
        time.sleep(random.uniform(2, 4))
        
    except Exception as e:
        print(f"查询失败: {name} - 原因: {e}")
        results.append(None) # 标记为空，方便之后检查
        time.sleep(60) # 如果报错（通常是频率限制），多休息一会

# 将结果填入表格
df['Average_Popularity_Score'] = results

# 保存最终文件
df.to_csv('Final_DWTS_Popularity.csv', index=False)
print("大功告成！文件已保存为 Final_DWTS_Popularity.csv")
