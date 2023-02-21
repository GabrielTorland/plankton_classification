mkdir dataset
echo "creating dataset"
py.exe ds_parser.py -r IHLS -c station_csv -d dataset > parse_log.txt
echo "compying abnormal data to an external folder for inspection"
py.exe ds_abnormal_bg.py -s dataset -c 1 --std 3
echo "plotting the dataset distribution"
py.exe ds_overview.py -s dataset

echo "creating directories"
mkdir organized_dataset
mkdir organized_dataset\\utils
mkdir organized_dataset\\logs
mkdir organized_dataset\\plots
echo "moving dataset"
move dataset organized_dataset\\dataset
echo "moving anomalies"
move anomalies organized_dataset\\anomalies
echo "moving files"
move anomaly_path.json organized_dataset\\utils\\anomaly_path.json
move background_dist.json organized_dataset\\utils\\anomaly_dist.json
move parse_log.txt organized_dataset\\logs\\parse_log.txt
move unkown_ids.txt organized_dataset\\logs\\unkown_ids.txt
move ds_percentage.txt organized_dataset\\logs\\ds_percentage.txt
move dist_pie_chart.png organized_dataset\\plots\\dist_pie_chart.png
move dist_pole.png organized_dataset\\plots\\dist_pole.png
move dist_train.png organized_dataset\\plots\\dist_train.png
move dist_test.png organized_dataset\\plots\\dist_test.png
move dist_val.png organized_dataset\\plots\\dist_val.png
