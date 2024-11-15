import os, random, argparse


def split_files_in_directory(directory_path, output_path, split_ratio=0.8):
    # 获取目录中的所有文件
    all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # 打乱文件顺序
    random.shuffle(all_files)

    # 计算分割点
    split_point = int(len(all_files) * split_ratio)

    # 分成两组
    group1_files = all_files[:split_point]
    group2_files = all_files[split_point:]

    # 创建输出文件路径
    output_file1 = os.path.join(output_path, "group1.txt")
    output_file2 = os.path.join(output_path, "group2.txt")

    # 将文件名写入输出文件
    with open(output_file1, "w") as f1:
        for file in group1_files:
            f1.write(file + "\n")

    with open(output_file2, "w") as f2:
        for file in group2_files:
            f2.write(file + "\n")

    print(f"Files have been split into {output_file1} and {output_file2}.")


def main():
    parser = argparse.ArgumentParser(description="Split files in a directory into two groups based on a ratio.")
    parser.add_argument("--img_path", required=True, type=str, help="Path to the directory containing the files to be split.")
    parser.add_argument("--split_path", required=True, type=str, help="Path to the directory where the output files will be saved.")
    parser.add_argument("--ratio", required=True, type=float, help="Ratio to split the files. For example, 0.8 means 80:20 split.")

    args = parser.parse_args()

    split_files_in_directory(args.img_path, args.split_path, args.ratio)


if __name__ == "__main__":
    main()
