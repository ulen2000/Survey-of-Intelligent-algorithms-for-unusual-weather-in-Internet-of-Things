from skimage import io, metrics
import csv
import argparse
import sys,os,re

parser = argparse.ArgumentParser(description="evaluator")
parser.add_argument("--algo", type=str, default="LPN", help='algorithm')
parser.add_argument("--dataset", type=str, default="Rain100H", help='dataset name')
parser.add_argument("--gt_path", type=str, default="/home/ubuntu/code/datasets/test/Rain100H/ground_truth", help='path to ground-truth results')
opt = parser.parse_args()


file_path = "/home/ubuntu/code/results/" + opt.dataset + "/" + opt.algo
print(file_path)
gt_path = opt.gt_path + "/"

headers = ['Image', 'PSNR', 'SSIM']

rows = []
for file_name in os.listdir(file_path):

	if opt.dataset == "Rain1400":
		test_file = os.path.join(file_path, file_name)
		test1 = io.imread(test_file)
		print(file_name)
		gt_file = gt_path + re.findall(r'(.+?)\_', file_name)[0] + ".jpg"
		print(gt_file)
		gt = io.imread(gt_file)
		psnr = metrics.peak_signal_noise_ratio(test1, gt, data_range=255)
		ssim = metrics.structural_similarity(test1, gt, data_range=255, multichannel=True)
		item = {"Image":file_name,"PSNR":psnr, "SSIM":ssim}
		print(item)
		rows.append(item)
	else :
		test_file = os.path.join(file_path, file_name)
		test1 = io.imread(test_file)
		print(file_name)
		if opt.dataset == "Rain12":
			gt_file = gt_path + file_name
		else:
			gt_file = gt_path + re.findall(r"o(.+?)\.", file_name)[0] + ".png"
		print(gt_file)
		gt = io.imread(gt_file)
		psnr = metrics.peak_signal_noise_ratio(test1, gt, data_range=255)
		ssim = metrics.structural_similarity(test1, gt, data_range=255, multichannel=True)
		item = {"Image": file_name, "PSNR": psnr, "SSIM": ssim}
		print(item)
		rows.append(item)


with open( opt.algo +'_quality_result.csv','w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)