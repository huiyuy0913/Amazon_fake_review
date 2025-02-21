import requests
import json
import pandas
import os
import shutil
from pprint import pprint


if not os.path.exists("scrape_json_files_oxylabs"):
    os.mkdir("scrape_json_files_oxylabs")

f = open("oxylabs_user","r")
oxylabs_user = f.read()
f.close()

f = open("oxylabs_pass","r")
oxylabs_pass = f.read()
f.close()

asin_list = pandas.read_csv("HK_also_view_no_category_bed_pillows_rest_before_90.csv")
# asin_list = pandas.read_csv("HK_also_view_no_category_final.csv")


for index,row  in asin_list.iterrows():
    asin = row["asin"]
    # set up the request parameters
    file_name = "scrape_json_files_oxylabs/asin_" + asin 
    if os.path.exists(file_name + ".json"):
        print("file exists", asin)
    else:
        try:
            print("downloading:",asin)
            payload = {
                'source': 'amazon_product',
                'domain': 'com',
                'query': asin,
                'parse': True,
                'context': [
                {
                'key': 'autoselect_variant', 'value': True
                }],
            }


            # Get response.
            response = requests.request(
                'POST',
                'https://realtime.oxylabs.io/v1/queries',
                auth=(oxylabs_user, oxylabs_pass),
                json=payload,
            )

            if response.status_code == 200:
                json_data = response.json()

                # Save the JSON data to a file
                json_file_path = f"{file_name}.json"
                with open(json_file_path, 'w') as json_file:
                    json.dump(json_data, json_file)

                print(f"JSON data for {asin} saved to {json_file_path}")
            else:
                print(f"Failed to get data for {asin}, status code: {response.status_code}")
        except Exception as e:
            print(e)

folder_path = 'scrape_json_files_oxylabs'
shutil.make_archive('scrape_json_files_oxylabs_bed_pillows_rest_before_90', 'zip', folder_path)
print("done")