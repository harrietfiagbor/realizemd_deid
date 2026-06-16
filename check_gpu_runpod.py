import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIG ---
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
# From your screenshot, this is your specific Pod ID
MY_POD_ID = "ze6wkd7g171o9w"
# The GPU type your pod uses
TARGET_GPU_ID = "NVIDIA RTX 2000 Ada Generation"


def resume_my_pod(api_key, pod_id):
    url = f"https://api.runpod.io/graphql?api_key={api_key}"

    # GraphQL mutation to resume a stopped pod
    mutation = """
    mutation ResumePod($input: PodResumeInput!) {
      podResume(input: $input) {
        id
        desiredStatus
        imageName
      }
    }
    """

    variables = {"input": {"podId": pod_id, "gpuCount": 1}}

    try:
        response = requests.post(
            url, json={"query": mutation, "variables": variables}, timeout=10
        )
        res = response.json()

        if "errors" in res:
            error_msg = res["errors"][0]["message"]
            print(f"❌ Resume Failed: {error_msg}")
            return False

        data = res.get("data", {}).get("podResume", {})
        if data.get("desiredStatus") == "RUNNING":
            print(f"🚀 SUCCESS! Pod {pod_id} is now starting up.")
            return True
        return False

    except Exception as e:
        print(f"⚠️ API Error during resume: {e}")
        return False


def monitor_and_resume(api_key, gpu_id, pod_id):
    url = f"https://api.runpod.io/graphql?api_key={api_key}"
    query = """
    query GetGpu($id: String!) {
      gpuTypes(input: { id: $id }) {
        lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
          stockStatus
          availableGpuCounts
        }
      }
    }
    """

    try:
        response = requests.post(
            url, json={"query": query, "variables": {"id": gpu_id}}, timeout=10
        )
        gpu_data = response.json().get("data", {}).get("gpuTypes", [])

        if gpu_data:
            stock = gpu_data[0].get("lowestPrice")
            count = stock.get("availableGpuCounts") or 0
            status = stock.get("stockStatus")

            print(
                f"[{time.strftime('%H:%M:%S')}] 🖥️  Checking... Stock: {status} | Avail: {count}"
            )

            # If there is even one GPU available, try to grab it for your pod
            if status not in [None, "OUT_OF_STOCK"] and count > 0:
                print(
                    "⚡ GPU Found! Sending Resume command for 'uneven_teal_chipmunk'..."
                )
                return resume_my_pod(api_key, pod_id)

    except Exception as e:
        print(f"Error checking stock: {e}")
    return False


# --- RUN ---
print(f"Watching for a spot to open for Pod: {MY_POD_ID}...")
while True:
    if monitor_and_resume(API_KEY, TARGET_GPU_ID, MY_POD_ID):
        # Notify with sound
        for _ in range(10):
            print("\a", end="", flush=True)
            time.sleep(0.2)
        break
    time.sleep(20)  # Check every 20 seconds
