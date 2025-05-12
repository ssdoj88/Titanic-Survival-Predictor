import os
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url, verify=False)

# Save the dataset
with open('data/titanic.csv', 'wb') as f:
    f.write(response.content)

print("Dataset downloaded successfully to data/titanic.csv") 