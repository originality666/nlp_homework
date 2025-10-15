# nlp_homework

# Step 1: 抓取样本
python scraper.py

# Step 2: 计算中文文本的熵（2M--6M）
python analyzer.py --input_dir collected_samples --out results/chinese_results.json --type chinese --scales 2000000 3000000 4000000 5000000 6000000

# Step 3: 计算英文文本的熵（2M--6M）
python analyzer.py --input_dir collected_samples --out results/english_results.json --type english --scales 2000000 3000000 4000000 5000000 6000000
