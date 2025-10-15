# Step 1: 抓取样本
python scraper.py

# Step 2: 计算英文文本
python analyzer.py --input_dir collected_samples --type english --scales 1000000 2000000 3000000 4000000 5000000 6000000 --out results_english.json

# Step 3: 计算中文文本
python analyzer.py --input_dir collected_samples --type chinese --scales 1000000 2000000 3000000 4000000 5000000 6000000 --out results_chinese.json
