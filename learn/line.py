import urllib.request
import os
import json
from time import sleep
import difflib
#from mymail import send_mail
import requests
import unicodedata
import csv

def notify_line(message):
    url = "https://notify-api.line.me/api/notify"
    access_token = 'NAxGP7MF0jek8iQsdGL6WOjzGUbmc04y59rCdKj3eYo'
    headers = {'Authorization': 'Bearer ' + access_token}
    payload = {'message': message}
    try:
        r = requests.post(url, headers=headers, params=payload,)
    except:
        print("LINE失敗")
        pass

def notify_selefplay_result():
    l = read_csv("selfplay_result.csv")
    
    result = ','.join(l[0][0:5])
    result += "\n"
    result += ','.join(l[1])
    notify_line(result)

def read_csv(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

if __name__ == '__main__':
    notify_selefplay_result()
