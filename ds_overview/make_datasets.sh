mkdir dataset
echo "creating dataset"
/usr/bin/python3 ds_parser.py -r IHLS -c station_csv -d dataset > parse_log.txt
echo "compying abnormal data to an external folder for inspection"
/usr/bin/python3 ds_abnormal_bg.py -s dataset -c 1 --std 3
echo "plotting the dataset distribution"
/usr/bin/python3 ds_overview.py -s dataset

echo "creating directories"
mkdir organized_dataset
mkdir organized_dataset/utils
mkdir organized_dataset/logs
mkdir organized_dataset/plots
echo "moving dataset"
mv dataset organized_dataset/dataset
echo "moving anomalies"
mv anomalies organized_dataset/anomalies
echo "moving files"
mv anomaly_path.json organized_dataset/utils/anomaly_path.json
mv background_dist.json organized_dataset/utils/background_dist.json
mv parse_log.txt organized_dataset/logs/parse_log.txt
mv unkown_ids.txt organized_dataset/logs/unkown_ids.txt
mv ds_percentage.txt organized_dataset/logs/ds_percentage.txt
mv dist_pie_chart.png organized_dataset/plots/dist_pie_chart.png
mv dist_pole.png organized_dataset/plots/dist_pole.png
mv dist_train.png organized_dataset/plots/dist_train.png
mv dist_test.png organized_dataset/plots/dist_test.png
mv dist_val.png organized_dataset/plots/dist_val.png
