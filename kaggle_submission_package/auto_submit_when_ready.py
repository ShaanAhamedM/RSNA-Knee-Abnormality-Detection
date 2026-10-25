import time
import requests
import subprocess
import os

token = 'KGAT_d875793e66b979178c0844895d5ca95d'
headers = {'User-Agent': 'kaggle-api/v1.7.4.5', 'Authorization': f'Bearer {token}'}

print("[AUTO-SUBMITTER] Monitoring shaanahamedm/rsna-knee-grandmaster-inference...")

start_time = time.time()
while time.time() - start_time < 900:  # 15 minutes max
    try:
        r = requests.get(
            'https://www.kaggle.com/api/v1/kernels/status?username=shaanahamedm&kernelSlug=rsna-knee-grandmaster-inference',
            headers=headers,
            timeout=10
        )
        data = r.json()
        st = data.get('status')
        print(f"[{time.strftime('%H:%M:%S')}] Kernel status: {st}")
        
        if st == 'complete':
            print("\n[AUTO-SUBMITTER] Kernel finished running successfully! Triggering competition submission now...")
            cmd = [
                r"D:\Miniconda3\envs\rsna_env\Scripts\kaggle.exe",
                "competitions", "submit",
                "rsna-knee-abnormality-detection",
                "-f", "submission.csv",
                "-k", "shaanahamedm/rsna-knee-grandmaster-inference",
                "-v", "2",
                "-m", "Phase 7 Grandmaster Multi-Backbone Ensemble"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print("Submit output:", res.stdout)
            print("Submit error:", res.stderr)
            if res.returncode == 0:
                print("SUCCESSFULLY SUBMITTED GRANDMASTER ENSEMBLE TO LEADERBOARD!")
                break
            else:
                # Try version 1 as fallback
                print("Retrying with version 1...")
                cmd[10] = "1"
                res1 = subprocess.run(cmd, capture_output=True, text=True)
                print("Fallback output:", res1.stdout, res1.stderr)
                break
        elif st == 'error':
            print("[WARN] Kernel completed with error:", data.get('failureMessage'))
            break
            
    except Exception as e:
        print(f"Poll check exception: {e}")
        
    time.sleep(10)

print("[AUTO-SUBMITTER] Done.")
