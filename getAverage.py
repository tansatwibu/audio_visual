import os

base_dir = "results2/"

arr = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

sdr, sdr_mixed, sdr_sdr_mixed, sir, sar = [], [], [], [], []


with open(os.path.join(base_dir, 'solo_solo.txt'), 'w') as f:
    for folder_name in arr:
        folder_path = os.path.join(base_dir, folder_name)
        try:
            with open(os.path.join(folder_path, 'eval.txt'), 'r') as output_file:
                line = output_file.readlines()[0].strip()
                metrics = line.split(" ")
                sdr.append(float(metrics[0]))
                sdr_mixed.append(float(metrics[1]))
                sdr_sdr_mixed.append(float(metrics[2]))
                sir.append(float(metrics[3]))
                sar.append(float(metrics[4]))

                f.write(f"{folder_name} | {line}\n")
        except Exception as e:
            print(f"Loi o thu muc {folder_name}: {e}")

    if len(sdr) > 0:
        avg_line = f"{sum(sdr)/len(sdr):.3f} {sum(sdr_mixed)/len(sdr_mixed):.3f} {sum(sdr_sdr_mixed)/len(sdr_sdr_mixed):.3f} {sum(sir)/len(sir):.3f} {sum(sar)/len(sar):.3f}"
        f.write(avg_line + '\n')
        print(f" Ket qua ghi tai {os.path.join(base_dir, 'solo_solo.txt')}")