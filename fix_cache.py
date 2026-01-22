import os
import glob

print("🔍 Cleaning MobileNetV2 cache...")

# Your exact paths
paths_to_clean = [
    r"C:\Users\hello\AppData\Roaming\Python\Python313\site-packages\.keras\models\*mobilenet*",
    r"C:\Users\hello\.keras\*mobilenet*",
    r"C:\Users\hello\AppData\Local\Temp\*mobilenet*"
]

deleted = 0
for pattern in paths_to_clean:
    for file in glob.glob(pattern):
        try:
            os.remove(file)
            print(f"🗑️  Deleted: {file}")
            deleted += 1
        except:
            pass

print(f"✅ Cleaned {deleted} files. Restart your train_model.py now!")
input("Press Enter when done...")
