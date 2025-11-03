import subprocess


REGION = 'europe-west2'
PROJECT_ID = subprocess.Popen(
    ['gcloud', 'config', 'get-value', 'project'],
    stdout=subprocess.PIPE
).stdout.read().decode('ascii').strip()
BUCKET_NAME = f'gs://{PROJECT_ID}-bucket-marketprediction'
PIPELINE_ROOT = f'{BUCKET_NAME}/pipeline_root_market/'
SERVICE_ACCOUNT = '703588886589-compute@developer.gserviceaccount.com'