# Install dependencies:
pip install -r requirements.txt



# Set your Google API key:
export GOOGLE_API_KEY="your-api-key"


# Embed the dataset:
python embed.py
# for powershell
$env:GOOGLE_API_KEY = "your-api-key"



# Run the app:
python main.py