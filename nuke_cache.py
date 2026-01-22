import os
import glob
import shutil

print("💣 NUCLEAR MOBILENET CACHE WIPE")

# AGGRESSIVE CLEANUP - ALL LOCATIONS
locations = [
    # Your Python313 keras cache
    r"C:\Users\hello\AppData\Roaming\Python\Python313\site-packages\.keras",
    
    # Standard keras cache  
    r"C:\Users\hello\.keras",
    
    # Temp files
    r"C:\Users\hello\AppData\Local\Temp",
    
    # OneDrive temp
    r"C:\Users\hello\OneDrive\Desktop\AI-projects\buffalo_project\*.h5"
]

for loc in locations:
    if os.path.exists(loc):
        if os.path.isdir(loc):
            for file in glob.glob(os.path.join(loc, "*mobilenet*")):
                try:
                    os.remove(file)
                    print(f"🗑️  {file}")
                except:
                    pass
        else:
            try:
                os.remove(loc)
                print(f"🗑️  {loc}")
            except:
                pass

print("✅ ALL CACHE DESTROYED")
print("🔄 RESTART TERMINAL & RUN train_model.py")
input("Press Enter after restart...")
