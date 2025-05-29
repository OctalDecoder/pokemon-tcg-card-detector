import os
import sys
from colorama import init, Fore, Style
import concurrent.futures

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from prototype_cnn import detect

# define your expected outputs
EXPECTED = {
    "1.png": ["A2b 44"],
    "2.png": ["A2b 36"],
    "3.png": ["A2b 14"],
    "4.png": ["A2b 34"],
    "5.png": ["A2b 92"],
    "7.png": ["A2b 17"],
    "Screenshot_2025.05.19_15.43.48.629.png": [],
    "Screenshot_2025.05.19_15.43.52.809.png": [],
    "Screenshot_2025.05.19_15.43.55.496.png": [],
    "Screenshot_2025.05.19_15.44.01.059.png": [],
    "Screenshot_2025.05.19_15.44.07.614.png": [],
    "Screenshot_2025.05.19_15.44.29.330.png": ["A2b 68", "A2b 38", "A2b 50", "A2b 59", "A2b 13"],
    "Screenshot_2025.05.19_15.44.51.594.png": ["A2b 66", "A2b 67", "A2b 68", "A2b 69", "A2b 70", "A1 195", "A2b 71", "A2b 72", "A2b 73", "A2b 75", "A2b 76"],
    "Screenshot_2025.05.27_11.25.43.406.png": ["A3 48", "A3 31", "A3 180", "A3 133", "A3 151", "A3 17", "A3 51", "A3 106", "A3 134", "A3 6", "A3 73", "A3 129"],
    "Screenshot_2025.05.27_11.25.52.233.png": ["A3 15", "A3 86", "A3 43", "A3 169", "A3 57", "A3 72", "A3 152", "A3 3", "A3 52", "A3 76", "A3 79", "A2b 8"],
    "Screenshot_2025.05.27_11.26.06.802.png": ["A2b 1", "A2b 2", "A2b 4", "A2b 5", "A2b 7", "A2b 6", "A2b 8", "A2b 9", "A2b 11", "A2b 12", "A2b 13", "A2b 14", "A2b 15", "A2b 16", "A2b 17", "A2b 18", "A2b 20", "A2b 21", "A2b 22", "A2b 23", "A2b 24", "A2b 25", "A2b 27", "A2b 28", "A2b 26", "A2b 29", "A2b 30"],
    "Screenshot_2025.05.27_11.26.11.871.png": ["A2b 21", "A2b 22", "A2b 23", "A2b 24", "A2b 25", "A2b 26", "A2b 27", "A2b 28", "A2b 29", "A2b 30", "A2b 32", "A2b 33", "A2b 34", "A2b 36", "A2b 37", "A2b 38", "A2b 39", "A2b 40", "A2b 41", "A2b 42", "A2b 43", "A2b 44", "A2b 45", "A2b 46", "A2b 47", "A2b 49", "A2b 50", "A2b 51", "A2b 52", "A2b 53", "A2b 54", "A2b 55"],
    "Screenshot_2025.05.27_11.26.17.795.png": ["A2b 46", "A2b 47", "A2b 49", "A2b 50", "A2b 51", "A2b 52", "A2b 53", "A2b 54", "A2b 55", "A2b 56", "A2b 57", "A2b 58", "A2b 59", "A2b 60", "A2b 61", "A2b 62", "A2b 63", "A2b 64", "A2b 65", "A2b 66", "A2b 67", "A2b 68", "A2b 69", "A2b 70", "A2b 71", "A2b 72", "A2b 73", "A2b 75", "A2b 76"],
    "Screenshot_20250519-170540_Pokmon_TCGP.jpg": ["A2b 60"],
    "Screenshot_20250519-170549_Pokmon_TCGP.jpg": ["A2b 47"],
    "Screenshot_20250519-170556_Pokmon_TCGP.jpg": ["A2b 63"],
    "Screenshot_20250519-170603_Pokmon_TCGP.jpg": ["A2b 35"],
    "Screenshot_20250519-170610_Pokmon_TCGP.jpg": ["A2b 35"],
    "Screenshot_20250526_224419_Pokmon_TCGP.jpg": ["A3 126"],
    "Screenshot_20250526_224424_Pokmon_TCGP.jpg": ["A3 85"],
    "Screenshot_20250526_224428_Pokmon_TCGP.jpg": ["A3 116"],
    "Screenshot_20250526_224432_Pokmon_TCGP.jpg": ["A3 13"],
    "Screenshot_20250526_224438_Pokmon_TCGP.jpg": ["A3 87"],
    "Screenshot_20250526_224511_Pokmon_TCGP.jpg": ["A3 35"],
    "Screenshot_20250526_224514_Pokmon_TCGP.jpg": ["A3 135"],
    "Screenshot_20250526_224517_Pokmon_TCGP.jpg": ["A3 73"],
    "Screenshot_20250526_224528_Pokmon_TCGP.jpg": ["A3 89"],
    "Screenshot_20250526_224532_Pokmon_TCGP.jpg": ["A3 142"],
    "Screenshot_20250526_224610_Pokmon_TCGP.jpg": ["A3 4"],
    "Screenshot_20250526_224614_Pokmon_TCGP.jpg": ["A3 14"],
    "Screenshot_20250526_224617_Pokmon_TCGP.jpg": ["A3 55"],
    "Screenshot_20250526_224620_Pokmon_TCGP.jpg": ["A3 50"],
    "Screenshot_20250526_224624_Pokmon_TCGP.jpg": ["A3 91"],
    "Screenshot_20250527_104722_Pokmon_TCGP.png": [],
    "Screenshot_20250527_104727_Pokmon_TCGP.png": [],
    "Screenshot_20250527_104735_Pokmon_TCGP.png": [],
    "Screenshot_20250527_104738_Pokmon_TCGP.png": [],
    "Screenshot_20250527_104744_Pokmon_TCGP.jpg": ["A3 125"],
    "Screenshot_20250527_104751_Pokmon_TCGP.jpg": ["A3 118"],
    "Screenshot_20250527_104754_Pokmon_TCGP.jpg": ["A3 125"],
    "Screenshot_20250527_104757_Pokmon_TCGP.jpg": ["A3 4"],
    "Screenshot_20250527_172047_Pokmon_TCGP.jpg": ["A2a 58", "A3 6", "A3 93", "A3 118", "A3 117"],
    "Screenshot_20250527_172100_Pokmon_TCGP.jpg": ["A3 93"],
    "Screenshot_20250527_231509_Pokmon_TCGP.jpg": ["A3 233"],
    "Screenshot_20250528_133930_Pokmon_TCGP.jpg": ["A1 155"],
    "Screenshot_20250528_133934_Pokmon_TCGP.jpg": ["A1 162"],
    "Screenshot_20250528_133938_Pokmon_TCGP.jpg": ["A1 60"],
    "Screenshot_20250528_134011_Pokmon_TCGP.jpg": ["A1 20"],
    "Screenshot_20250528_134013_Pokmon_TCGP.jpg": ["A1 105"],
    "Screenshot_20250528_134017_Pokmon_TCGP.jpg": ["A1 59"],
    "Screenshot_20250528_134019_Pokmon_TCGP.jpg": ["A1 27"],
    "Screenshot_20250528_134021_Pokmon_TCGP.jpg": ["A1 163"],
    "Screenshot_20250528_134038_Pokmon_TCGP.jpg": ["A1 51", "A1 52", "A1 53", "A1 59", "A1 60", "A1 74"],
    "Screenshot_20250528_172708_Pokmon_TCGP.jpg": ["A2 9"],
    "Screenshot_20250528_172711_Pokmon_TCGP.jpg": ["A2 44"],
    "Screenshot_20250528_172714_Pokmon_TCGP.jpg": ["A2 25"],
    "Screenshot_20250528_172716_Pokmon_TCGP.jpg": ["A2 63"],
    "Screenshot_20250528_172719_Pokmon_TCGP.jpg": ["A2 43"],
    "Screenshot_20250528_172752_Pokmon_TCGP.jpg": ["A2 80"],
    "Screenshot_20250528_172755_Pokmon_TCGP.jpg": ["A2 94"],
    "Screenshot_20250528_172757_Pokmon_TCGP.jpg": ["A2 16"]
}

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


def evaluate(results):
    tp = fp = fn = 0
    for fname, expected in EXPECTED.items():
        got = results.get(fname, [])
        expected_set = set(expected)
        got_set = set(got)
        tp += len(expected_set & got_set)
        fp += len(got_set - expected_set)
        fn += len(expected_set - got_set)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_default():
    init(autoreset=True)

    # Verify images directory
    if not os.path.isdir(IMAGES_DIR):
        print(Fore.RED + f"Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    # Run detector
    results = detect(screenshot_dir=IMAGES_DIR, save_results_images=False, logging=False)
    if not isinstance(results, dict):
        print(Fore.RED + "Error: run() did not return a dict of results")
        sys.exit(1)

    errors = 0
    warnings_count = 0
    warnings_details = []
    errors_details = []

    for filename, expected_labels in EXPECTED.items():
        got_labels = results.get(filename, [])
        missing = set(expected_labels) - set(got_labels)
        extra = set(got_labels) - set(expected_labels)

        if extra:
            print(Fore.RED + f"ERROR: {filename}")
            print(Fore.RED + f"  Unexpected labels: {sorted(extra)}")
            errors += 1
            errors_details.append((filename, sorted(extra)))
        elif missing:
            print(Fore.YELLOW + f"WARNING: {filename}")
            print(Fore.YELLOW + f"  Missing labels: {sorted(missing)}")
            warnings_count += 1
            warnings_details.append((filename, sorted(missing)))
        else:
            print(Fore.GREEN + f"PASS: {filename}")

    # Summary
    print(Style.BRIGHT + "\nScreenshot Summary:")
    passed = len(EXPECTED) - errors - warnings_count
    print(Fore.GREEN + f"  Passed:      {passed}")
    print(Fore.YELLOW + f"  Missing:    {warnings_count}")
    print(Fore.RED + f"  Incorrect:     {errors}")

    # Detailed report
    total = 0
    if warnings_details:
        print(Style.BRIGHT + "\nMissing Detections:")
        for fname, missing in warnings_details:
            total += len(missing)
            print(Fore.YELLOW + f"  {fname}: missing {missing}")
        print(Fore.YELLOW + f"  Total: {total}")

    total = 0
    if errors_details:
        print(Style.BRIGHT + "\nIncorrect Classifications:")
        for fname, extra in errors_details:
            total += len(extra)
            print(Fore.RED + f"  {fname}: unexpected {extra}")
        print(Fore.RED + f"  Total: {total}")

    # exit code
    if errors > 0:
        sys.exit(1)
    sys.exit(0)


def _evaluate_thresh(args):
    c, y = args
    results = detect(
        screenshot_dir=IMAGES_DIR,
        save_results_images=False,
        logging=False,
        cnn_thresh=c,
        yolo_thresh=y
    )
    metrics = evaluate(results)
    return c, y, metrics


def tune(cnn_range, yolo_range):
    init(autoreset=True)
    combos = [(c, y) for c in cnn_range for y in yolo_range]
    best = {"f1": -1.0, "cnn_thresh": None, "yolo_thresh": None}
    print(Style.BRIGHT + "Tuning thresholds with multithreading...")

    # use max workers equal to CPU cores
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_evaluate_thresh, combo): combo for combo in combos}
        for future in concurrent.futures.as_completed(futures):
            c, y, metrics = future.result()
            print(f"cnn={c:.2f}, yolo={y:.2f} → P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            if metrics['f1'] > best['f1']:
                best.update({"f1": metrics['f1'], "cnn_thresh": c, "yolo_thresh": y})

    print(Style.BRIGHT + "\nBest thresholds:")
    print(Fore.CYAN + f"  CNN thresh:  {best['cnn_thresh']:.2f}")
    print(Fore.CYAN + f"  YOLO thresh: {best['yolo_thresh']:.2f}")
    print(Fore.CYAN + f"  F1:          {best['f1']:.3f}")


if __name__ == "__main__":
    if "--tune" in sys.argv:
        cnn_vals = [i/100 for i in range(5, 95, 5)]
        yolo_vals = [i/100 for i in range(5, 85, 10)]
        tune(cnn_vals, yolo_vals)
    else:
        run_default()

