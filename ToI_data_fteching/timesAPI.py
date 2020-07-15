import requests
import json

url = "https://devru-times-of-india.p.rapidapi.com/feeds/feedurllist.cms"

querystring = {"catagory":"Latest"}

headers = {
    'x-rapidapi-host': "devru-times-of-india.p.rapidapi.com",
    'x-rapidapi-key': "64f2144c35msh02e2da58913da29p14effajsn493a2f7645ce"
    }

response = requests.request("GET", url, headers=headers, params=querystring)
f=response.text
s={}
s["item"]=f
g=s["item"]
res = json.loads(g) 
for l in res["Item"]:
	if(l["name"]=="Sports"):
		print(l)
