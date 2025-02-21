import requests
import json
import pandas
import os
# import shutil

if not os.path.exists("scrape_json_files_oxylabs_bed_pillows_rest_last_90"):
    os.mkdir("scrape_json_files_oxylabs_bed_pillows_rest_last_90")

f = open("api_key_rainforest","r")
api_key_rainforest = f.read()
f.close()

asin_list = pandas.read_csv("HK_also_view_no_category_bed_pillows_rest_last_90.csv")


for index,row  in asin_list.iterrows():
    asin = row["asin"]
    # set up the request parameters
    file_name = "scrape_json_files_oxylabs_bed_pillows_rest_last_90/asin_" + asin 
    if os.path.exists(file_name + ".json"):
        print("file exists", asin)
    else:
        try:
            print("downloading:",asin)
            params = {
            'api_key': api_key_rainforest,
            'amazon_domain': 'amazon.com',
            'asin': asin,
            'type': 'product'
            }

            # make the http GET request to Rainforest API
            api_result = requests.get('https://api.rainforestapi.com/request', params)

            # print the JSON response from Rainforest API
            # print(json.dumps(api_result.json()))

            if api_result.status_code == 200:
                json_data = api_result.json()

                # Save the JSON data to a file
                json_file_path = f"{file_name}.json"
                with open(json_file_path, 'w') as json_file:
                    json.dump(json_data, json_file)

                print(f"JSON data for {asin} saved to {json_file_path}")
            else:
                print(f"Failed to get data for {asin}, status code: {api_result.status_code}")
        except Exception as e:
            print(e)

folder_path = 'scrape_json_files_oxylabs_bed_pillows_rest_last_90'
# shutil.make_archive('HK_also_view_no_category_final.csv', 'zip', folder_path)
print("done")