from google.cloud import bigquery
from google.oauth2 import service_account

CREDENTIAL_PATH = "..\secret.json"
credentials = service_account.Credentials.from_service_account_file(CREDENTIAL_PATH,scopes=["https://www.googleapis.com/auth/cloud-platform"],)
CLIENT = bigquery.Client(credentials=credentials, project=credentials.project_id,)