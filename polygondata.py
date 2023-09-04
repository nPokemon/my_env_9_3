import requests
import datetime
import json

# Set up API key and parameters
api_key = 'pYhkqqWAWgIdSUgW0DciumhD5BPLzdWU'
symbol = 'BTC'
interval = '5'
time = "minute"
start_date = '2020-01-01'
end_date = datetime.date.today().strftime('%Y-%m-%d')

# Construct API request URL
url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/{time}/{start_date}/{end_date}?apiKey={api_key}'

# Make API request
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Parse JSON response
    data = response.json()

    try:
        results = data['results']
        # Your code to process the results goes here

        # Construct a dynamic CSV filename based on the symbol and date
        file_name = f'{symbol}_data_{start_date}_to_{end_date}.csv'

        # Save to CSV file with the dynamic filename
        with open(file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header row
            header = results[0].keys()
            csv_writer.writerow(header)

            # Write data rows
            for row in results:
                csv_writer.writerow(row.values())

        print(f'Data extracted successfully and saved as {file_name}!')
    except KeyError as e:
        print(f"KeyError: {e}. Check the structure of the API response.")
else:
    print('Request failed with status code:', response.status_code)
