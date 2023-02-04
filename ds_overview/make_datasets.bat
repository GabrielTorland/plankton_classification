



:: dataset including unique pre-processed images
mkdir dataset
echo "creating dataset"
py.exe ds_parser.py -r IHLS -c station_csv -d dataset > parse_log.txt
echo "compying abnormal data to an external folder for inspection"
py.exe ds_abnormal_bg.py -s dataset -c 1
echo "plotting the dataset distribution"
py.exe ds_overview.py -s dataset

echo "creating directories"
mkdir processed_dataset
mkdir processed_dataset\\utils
mkdir processed_dataset\\logs
mkdir processed_dataset\\plots
echo "moving dataset"
move dataset processed_dataset\\dataset
echo "moving anomalies"
move anomalies processed_dataset\\anomalies
echo "moving files"
move anomaly_path.json processed_dataset\\utils\\anomaly_path.json
move background_dist.json processed_dataset\\utils\\anomaly_dist.json
move parse_log.txt processed_dataset\\logs\\parse_log.txt
move unkown_ids.txt processed_dataset\\logs\\unkown_ids.txt
move ds_percentage.txt processed_dataset\\logs\\ds_percentage.txt
move dist_pie_chart.png processed_dataset\\plots\\dist_pie_chart.png
move dist_pole.png processed_dataset\\plots\\dist_pole.png
move dist_train.png processed_dataset\\plots\\dist_train.png
move dist_test.png processed_dataset\\plots\\dist_test.png
move dist_val.png processed_dataset\\plots\\dist_val.png

:: raw dataset
mkdir dataset
echo "creating dataset"
py.exe ds_parser.py -r IHLS -c station_csv -d dataset -p 0 > parse_log.txt
echo "compying abnormal data to an external folder for inspection"
py.exe ds_abnormal_bg.py -s dataset -c 1
echo "plotting the dataset distribution"
py.exe ds_overview.py -s dataset

echo "creating directories"
mkdir raw_dataset
mkdir raw_dataset\\utils
mkdir raw_dataset\\logs
mkdir raw_dataset\\plots
echo "moving dataset"
move dataset raw_dataset\\dataset
echo "moving anomalies"
move anomalies raw_dataset\\anomalies
echo "moving files"
move anomaly_path.json raw_dataset\\utils\\anomaly_path.json
move background_dist.json raw_dataset\\utils\\anomaly_dist.json
move parse_log.txt raw_dataset\\logs\\parse_log.txt
move unkown_ids.txt raw_dataset\\logs\\unkown_ids.txt
move ds_percentage.txt raw_dataset\\logs\\ds_percentage.txt
move dist_pie_chart.png raw_dataset\\plots\\dist_pie_chart.png
move dist_pole.png raw_dataset\\plots\\dist_pole.png
move dist_train.png raw_dataset\\plots\\dist_train.png
move dist_test.png raw_dataset\\plots\\dist_test.png
move dist_val.png raw_dataset\\plots\\dist_val.png
