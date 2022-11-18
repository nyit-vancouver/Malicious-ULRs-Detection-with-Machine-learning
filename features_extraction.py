#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from urllib.parse import urlparse
import bs4
import pandas as pd
import re
from bs4 import BeautifulSoup
import whois
import urllib
import numpy as np
import joblib
import csv
from googlesearch import search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
#import urllib.request
#import urllib2.urlopen
from urllib.request import urlopen
#from urllib2 import urlopen
import requests
from datetime import datetime
from tld import get_tld, get_fld
import warnings
warnings.filterwarnings("ignore")


# In[3]:


urldata = pd.read_csv('finaldata.csv')


# In[5]:


def getDomain(url):
    domain = urlparse(url).netloc
    if re.match(r"^www.",domain):
        domain = domain.replace("www.","")
    return domain

def havingIP(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

def haveAtSign(url):
    if "@" in url:
        at = 1
    else:
        at = 0
    return at

# 4 Length of URL (URL_Length)
def getLength(url):
    if len(url) < 54:
        length = 0
    else:
        length = 1
    return length

# 5 Depth of URL - Gives number of '/' in URL (URL_Depth)
def getDepth(url):
    s = urlparse(url).path.split('/')
    depth = 0
    for j in range(len(s)):
        if len(s[j]) != 0:
            depth = depth+1
    return depth

# 6 Redirection "//" in URL (Redirection)
def redirection(url):
    pos = url.rfind('//')
    if pos > 6:
        if pos > 7:
            return 1
        else:
            return 0
    else:
        return 0

# 7 "http/https" in Domain name (https_Domain)
def httpDomain(url):
    domain = urlparse(url).netloc
    if 'https' in domain:
        return 0
    else:
        return 1

def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1
    else:
        return 0

# 10 Web Traffic
def web_traffic(url):
    try:
    #Filling the whitespaces in the URL if any
        url = urllib.parse.quote(url)
        rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find(
        "REACH")['RANK']
        rank = int(rank)
    except TypeError:
        return 1
    if rank <100000:
        return 0
    else:
        return 1

def domainAge(url):
    try:
        domain_name = whois.whois(urlparse(url).netloc)
    except:
        return 1
    creation_date = domain_name.creation_date
    expiration_date = domain_name.expiration_date
    if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
        try:
            creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
            expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
        except:
            return 1
    if ((expiration_date is None) or (creation_date is None)):
        return 1
    elif ((type(expiration_date) is list) or (type(creation_date) is list)):
        return 1
    else:
        ageofdomain = abs((expiration_date - creation_date).days)
        if ((ageofdomain/30) < 6):
            age = 1
        else:
            age = 0
    return age

def domainEnd(url):
    try:
        domain_name = whois.whois(urlparse(url).netloc)
    except:
        return 1
    expiration_date = domain_name.expiration_date
    if isinstance(expiration_date,str):
        try:
            expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
        except:
            return 1
    if (expiration_date is None):
        return 1
    elif (type(expiration_date) is list):
        return 1
    else:
        today = datetime.now()
        end = abs((expiration_date - today).days)
        if ((end/30) < 6):
            end = 1
        else:
            end = 0
    return end

def iframe(response):
    if response == "":
        return 1
    else:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            return 0
        else:
            return 1

def mouseOver(response):
    if response == "" :
        return 1
    else:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            return 1
        else:
            return 0

def rightClick(response):
    if response == "":
        return 1
    else:
        if re.findall(r"event.button ?== ?2", response.text):
            return 0
        else:
            return 1

def forwarding(response):
    if response == "":
        return 1
    else:
        if len(response.history) <= 2:
            return 0
        else:
            return 1

def get_email(url):
    try:
        domain_name = whois.whois(urlparse(url).netloc)
        email = domain_name.emails
        if (email is None):
            flag = 1
        else:
            flag = 0
    except:
        flag = 1
    return flag

def google_index(url):
    site = search(url, 5)
    return 0 if site else 1

def achieve_subdomain(url):
    if havingIP(url)==0:
        try:
            return get_tld(url,as_object=True).subdomain
        except:
            return None

def achieve_tld(url):
    if havingIP(url)==0:
        try:
            return get_tld(url,as_object=True).tld
        except:
            return None

def achieve_fld(url):
    if havingIP(url)==0:
        try:
            return get_fld(url,as_object=True)
        except:
            return None

def url_length(url):
    return len(url)

def path_length(url):
    return len(urlparse(url).path)

def hostname_length(url):
    return len(urlparse(url).netloc)

def alpha_count(url):
    alpha = 0
    for i in url:
        if i.isalpha():
            alpha += 1
    return alpha

def digit_count(url):
    digits = 0
    for i in url:
        if i.isdigit():
            digits = digits + 1
    return digits

def count1(url):
    return url.count('.')

def count2(url):
    return url.count('@')

def count3(url):
    return url.count('-')

def count4(url):
    return url.count('%')

def count5(url):
    return url.count('?')

def count6(url):
    return url.count('=')

def num_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def first_dir_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0


# In[24]:


# Function to extract features
def featureExtraction(url):
    features = []
    features.append(havingIP(url))
    features.append(haveAtSign(url))
    features.append(getLength(url))
    features.append(getDepth(url))
    features.append(redirection(url))
    features.append(httpDomain(url))
    features.append(prefixSuffix(url))
    features.append(domainAge(url))
    features.append(domainEnd(url))
    '''try:
        response = requests.get(url)
    except:
        response = ""
    features.append(iframe(response))
    features.append(mouseOver(response))
    features.append(rightClick(response))
    features.append(forwarding(response))'''
    features.append(get_email(url))
    features.append(google_index(url))
    features.append(web_traffic(url))

    features.append(len(str(achieve_subdomain(url))))
    features.append(len(str(achieve_tld(url))))
    features.append(len(str(achieve_tld(url))))
    features.append(path_length(url))
    features.append(hostname_length(url))
    features.append(alpha_count(url))
    features.append(digit_count(url))
    features.append(url_length(url) - (alpha_count(url) + digit_count(url)))
    features.append(count1(url))
    features.append(count2(url))
    features.append(count3(url))
    features.append(count4(url))
    features.append(count5(url))
    features.append(count6(url))
    features.append(num_of_dir(url))
    features.append(first_dir_length(url))
    features.append(alpha_count(url) / url_length(url))
    features.append(digit_count(url) / url_length(url))
    features.append((url_length(url) - (alpha_count(url) + digit_count(url))) / url_length(url))

    return features


# In[46]:


# Create the dataframe to save the features
result = pd.DataFrame(columns=['use_of_ip', 'have_@', 'url_length', 'dir_depth', 'is_redirection',
       'is_https', 'have_-', 'domain_Age', 'domain_End', 'email',
       'google_index', 'web_traffic', 'subdomain_len', 'tld_len', 'fld_len',
       'url_path_len', 'hostname_len', 'url_alphas', 'url_digits', 'url_puncs',
       'count.', 'count@', 'count-', 'count%', 'count?', 'count=',
       'count_dirs', 'first_dir_len', 'pc_alphas', 'pc_digits', 'pc_puncs',
       'result'], index=[0])


# In[48]:


with open('finaldata.csv',encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]


# In[49]:


# extract the features of data
for i in range(len(urldata)):
    url = rows[i][0]
    try:
        #print(url)
        feature = featureExtraction(url)
        
        result.loc[i-1,['use_of_ip', 'have_@', 'url_length', 'dir_depth', 'is_redirection',
       'is_https', 'have_-', 'domain_Age', 'domain_End', 'email',
       'google_index', 'web_traffic', 'subdomain_len', 'tld_len', 'fld_len',
       'url_path_len', 'hostname_len', 'url_alphas', 'url_digits', 'url_puncs',
       'count.', 'count@', 'count-', 'count%', 'count?', 'count=',
       'count_dirs', 'first_dir_len', 'pc_alphas', 'pc_digits', 'pc_puncs']]=feature[0:31]
        #print(feature)
        time.sleep(2)
    except Exception:
        pass


# In[51]:


for i in range(len(result)):
    result['result'][i]=urldata['result'][i]


# In[64]:


# Transform data types
for item in ['use_of_ip', 'have_@', 'url_length', 'dir_depth', 'is_redirection',
       'is_https', 'have_-', 'domain_Age', 'domain_End', 'email',
       'google_index', 'web_traffic', 'subdomain_len', 'tld_len', 'fld_len',
       'url_path_len', 'hostname_len', 'url_alphas', 'url_digits', 'url_puncs',
       'count.', 'count@', 'count-', 'count%', 'count?', 'count=',
       'count_dirs', 'first_dir_len', 'result']:
    result[item]=result[item].astype(np.int64)

for item in ['pc_alphas', 'pc_digits', 'pc_puncs']:
    result[item]=result[item].astype(np.float64)
    


# In[151]:


# Save the result
result.to_csv('result.csv',index=False)

