import queue
import requests, json, time
import time
import psutil
import pandas as pd
from statistics import mean
from scipy import stats
import os.path
import csv
import tracemalloc
import threading
import numpy as np
import concurrent.futures

def query(url, method, payload=None):
    res = requests.Response()
    elapsed_time = 0
    if method == "GET":
        start_time = time.time()
        res = requests.get(url)
        elapsed_time = time.time() - start_time
    if method == "POST":
        start_time = time.time()
        res = requests.post(
            url,
            data=json.dumps(payload),
            #data = payload,
            headers={"content-type": "application/json"},
        )
        elapsed_time = time.time() - start_time
    res_content = None

    if res.content is not None:
        res_content = json.loads(res.content.decode("utf8"))
        #res_content = res.content

    return res_content, elapsed_time

#Choose model from [pac, mnb, svc, dt, kn, rf, log, mlpc]
#Type pac for PA Classifier OR mnb for Multinomial NB OR svc for Support Vector Classifier OR \ndt for Decision Tree OR kn for K Neighbors Classifier OR rf for Random Forest Classifier OR \nlog for Logistic Regression OR mlpc for Multilayer Perceptron Classifier (Neural Network)
#modell = ['pac', 'mnb', 'log', 'svc', 'dt', 'kn', 'rf', 'mlpc']
#modell = ['mnb', 'mlpc']

#choose number of partitions you desire, choose 1 if your desired learning is not distributed learning
#narr = [1, 5, 10, 20, 30]
#narr = [10]

model = 'mnb'
n = 10                      #number of partitions
root_url = "http://127.0.0.1:8080/"
index_es = "function"

#The below parameters (data_path, x_col, y_col, model and n) would be commented if you prefer to enter the parameters at the prompt
data_path = "news.csv"
x_col = "text"         # the independent variable column
y_col = "label"     # the label column

df = pd.read_csv(data_path, nrows=315)              #read the data with pandas
df = df.dropna()

dataarr = []                #empty array for the partitions

print('started for ', model, ' and number of partitions: ', n)
#to use the same data partition as the whole partitions
for i in range(n):
    dataarr.append(df)

ramusagearr = []                #ram usage empty array

def querycall(partition):
    # first extract the elements of x_col and y_col in dictionary formats that would be put into the query function as parameters
    x_keyarr = []
    y_keyarr = []
    partition_data = partition
    index = partition_data.index
    for j in index:
        j = str(j)
        x_keyarr.append(j)
        y_keyarr.append(j)
    # dictionaries of x and y with None as values for now
    x_dict = dict(zip(x_keyarr, [None] * len(x_keyarr)))
    y_dict = dict(zip(y_keyarr, [None] * len(y_keyarr)))
    for k in index:
        # now append the right values to the dictionaries of x and y to replace the initial None values
        x_dict[str(k)] = partition_data[x_col][k]
        y_dict[str(k)] = partition_data[y_col][k]
    input = {"x_dict": x_dict,
             "y_dict": y_dict,
             "model": model,
             "dst":"none"}
    # call the query function to trigger/invoke the serverless function on each partition
    response_content = query(url=root_url + index_es + "/payload-echo",
                             method="POST",
                             payload=input)
    return response_content

# to measure bandwidth
net1_out_1 = psutil.net_io_counters().bytes_sent  # bytes sent at the beginning
net1_in_1 = psutil.net_io_counters().bytes_recv  # bytes received at the beginning
#start tracemalloc library to get ram usage in bytes
tracemalloc.start()
# to measure time
# time at the beginning
t0 = time.time()

#FIRST METHOD
ex = concurrent.futures.ThreadPoolExecutor()
results = ex.map(querycall, dataarr)
real_results = list(results)

'''#SECOND METHOD
que = queue.Queue()
threads = list()
for ith in range(len(dataarr)):
    x = threading.Thread(target=lambda q, arg1: q.put(querycall(arg1)), args=(que, dataarr[ith]))
    threads.append(x)
    x.start()
# Join all the threads
for t in threads:
    t.join()'''

'''#THIRD METHOD
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    for ith in range(len(dataarr)):
        futures.append(executor.submit(querycall, partition=dataarr[ith]))
    for ith in range(len(futures)):
        result = futures[ith].result()'''

'''#FOURTH METHOD
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(querycall, partition) for partition in dataarr]
results = [f.result() for f in futures]'''

t1 = time.time()  # time at the end
current_usage, peak_usage = tracemalloc.get_traced_memory()
ramusagearr.append(current_usage)
# stopping the tracemalloc library
tracemalloc.stop()

# CPU seconds elapsed (floating point)
t1_t0 = t1 - t0
t1t0 = '%.3f' % (t1_t0)
# Get new net in/out
net2_out_1 = psutil.net_io_counters().bytes_sent  # bytes sent at the end
net2_in_1 = psutil.net_io_counters().bytes_recv  # bytes received at the end
# Compare and get current speed
if net1_in_1 > net2_in_1:
    current_in = 0
else:
    current_in = (net2_in_1 - net1_in_1)
if net1_out_1 > net2_out_1:
    current_out = 0
else:
    current_out = (net2_out_1 - net1_out_1)
network_1 = {"traffic_in": current_in, "traffic_out": current_out}
#total traffic
totaltraffic = current_in + current_out

sumcurrentusage = sum(ramusagearr)
print('memory usage', sumcurrentusage)

receivespeed = '%.3f' % (current_in / t1_t0)
sendspeed = '%.3f' % (current_out / t1_t0)

print("time elapsed: ", t1t0, "secs")
print("total traffic: ", totaltraffic, "bytes")
print("speed of inflow traffic: ", receivespeed, "bytes/sec")
print("speed of outflow traffic: ", sendspeed, "bytes/sec")
print('finished')
print(" ")


#WRITING THE DATA VALUES TO A CSV FILE
#the columns of the csv file
columns = ['model', 'number of partitions', 'time elapsed (secs)',
           'total traffic (bytes)', 'inflow speed (bytes/sec)', 'outflow speed (bytes/sec)',
           'memory usage (bytes)']
#check if the csv file already exists
if os.path.exists('results_payloadecho.csv') == True:
    with open('results_payloadecho.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        #writer.writeheader()
        if model == 'pac':
            datapar1 = {'model':'pac', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar1)
        if model == 'log':
            datapar2 = {'model':'log', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar2)
        if model == 'mnb':
            datapar3 = {'model':'mnb', 'number of partitions': n, 'time elapsed (secs)': t1t0, 'total traffic (bytes)': totaltraffic,
                        'inflow speed (bytes/sec)': receivespeed, 'outflow speed (bytes/sec)': sendspeed,
                        'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar3)
        if model == 'mlpc':

            datapar4 = {'model':'mlpc', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar4)
        if model == 'svc':
            datapar5 = {'model':'svc', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar5)
        if model == 'dt':
            datapar6 = {'model':'dt', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar6)
        if model == 'kn':
            datapar7 = {'model':'kn', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar7)
        if model == 'rf':
            datapar8 = {'model':'rf', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar8)
else:
    with open('results_payloadecho.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        if model == 'pac':
            datapar1 = {'model':'pac', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar1)
        if model == 'log':
            datapar2 = {'model':'log', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar2)
        if model == 'mnb':
            datapar3 = {'model':'mnb', 'number of partitions': n, 'time elapsed (secs)': t1t0, 'total traffic (bytes)': totaltraffic,
                        'inflow speed (bytes/sec)': receivespeed, 'outflow speed (bytes/sec)': sendspeed,
                        'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar3)
        if model == 'mlpc':

            datapar4 = {'model':'mlpc', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar4)
        if model == 'svc':
            datapar5 = {'model':'svc', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar5)
        if model == 'dt':
            datapar6 = {'model':'dt', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar6)
        if model == 'kn':
            datapar7 = {'model':'kn', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar7)
        if model == 'rf':
            datapar8 = {'model':'rf', 'number of partitions': n, 'time elapsed (secs)': t1t0,
                        'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                        'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
            writer.writerow(datapar8)
