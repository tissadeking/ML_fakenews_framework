import requests, json, time
import time
import psutil
import pandas as pd
from statistics import mean
from scipy import stats
import os.path
import csv
import tracemalloc

def query(url, method, payload=None):
    res = requests.Response()
    if method == "GET":
        res = requests.get(url)
    if method == "POST":
        res = requests.post(
            url,
            data=json.dumps(payload),
            headers={"content-type": "application/json"},
        )

    res_content = None

    if res.content is not None:
        res_content = json.loads(res.content.decode("utf8"))

    return res_content

#Choose model from [pac, mnb, svc, dt, kn, rf, log, mlpc]
#Type pac for PA Classifier OR mnb for Multinomial NB OR svc for Support Vector Classifier OR \ndt for Decision Tree OR kn for K Neighbors Classifier OR rf for Random Forest Classifier OR \nlog for Logistic Regression OR mlpc for Multilayer Perceptron Classifier (Neural Network)
#modell = ['pac', 'mnb', 'log', 'svc', 'dt', 'kn', 'rf', 'mlpc']
modell = ['pac']

#choose number of partitions you desire, choose 1 if your desired learning is not distributed learning
#narr = [1, 5, 10, 20, 30]
narr = [10]
for i in range(len(modell)):
    model = modell[i]
    for j in range(len(narr)):
        n = narr[j]

#if __name__ == "__main__":
        root_url = "http://127.0.0.1:8080/"
        index_es = "function"

        #The below parameters (data_path, x_col, y_col, model and n) would be commented if you prefer to enter the parameters at the prompt
        data_path = "news.csv"
        x_col = "text"         # the independent variable column
        y_col = "label"     # the label column

        df = pd.read_csv(data_path, nrows=3150)              #read the data with pandas
        df = df.dropna()
        # TO CREATE PARTITIONS OF THE DATA
        # Create bin edges (the points of divisions into partitions)
        step_size = int(len(df) / n)                        #number of data points belonging to each partition
        bin_edges = list(range(0, len(df), step_size))
        l = len(df)
        dataarr = []                                        #create empty array to hold the various partitions later
        #now put the partitions into the empty array - dataarr
        for i in range(n):
            if i == n - 1:
                partition = df[bin_edges[i]:l + 1]
                dataarr.append(partition)
            elif i <= n - 1:
                partition = df[bin_edges[i]:bin_edges[i + 1]]
                dataarr.append(partition)

        print('started for ', model, ' and number of partitions: ', n)

        # to measure time
        t0 = time.time()                                        #time at the beginning
        # to measure bandwidth
        net1_out = psutil.net_io_counters().bytes_sent          #bytes sent at the beginning
        net1_in = psutil.net_io_counters().bytes_recv           #bytes received at the beginning

        accuracyarr = []                                        #empty array to hold accuracies of various partitions later
        ramusagearr = []
        testaccuracyarr = []

        #Create dictionary to store the training parameters for all the partitions later
        keyarr = []
        for i in range(len(dataarr)):
            i = str(i)
            keyarr.append(i)
        parameterdict = dict(zip(keyarr, [None] * len(keyarr)))             #each value of the dictionary is None for now

        #Dataarr contains the whole partitions, call the query function, get accuracy and parameters for each of the partitions in dataarr
        for i in range(len(dataarr)):
            # start tracemalloc library to get ram usage in bytes
            tracemalloc.start()
            #first extract the elements of x_col and y_col in dictionary formats that would be put into the query function as parameters
            x_keyarr = []
            y_keyarr = []
            partition_data = dataarr[i]
            index = partition_data.index
            for j in index:
                j = str(j)
                x_keyarr.append(j)
                y_keyarr.append(j)
            #dictionaries of x and y with None as values for now
            x_dict = dict(zip(x_keyarr, [None] * len(x_keyarr)))
            y_dict = dict(zip(y_keyarr, [None] * len(y_keyarr)))
            for k in index:
                #now append the right values to the dictionaries of x and y to replace the initial None values
                x_dict[str(k)] = partition_data[x_col][k]
                y_dict[str(k)] = partition_data[y_col][k]

            input = {"x_dict": x_dict,
                     "y_dict": y_dict,
                      "X": x_col,
                      "y": y_col, "model": model}
            #call the query function to trigger/invoke the serverless function on each partition
            response_content = query(url=root_url + index_es + "/fakenews",
                method="POST",
                payload = input)
            print('for partition ', i+1, ': ', response_content)        #response for each partition
            accuracyarr.append(response_content['accuracy (validation)'])            #append the accuracy for each partition to the initial empty accuracy array
            testaccuracyarr.append(response_content['accuracy (test)'])
            #append the parameters for each partition to the parameter dictionary declared earlier
            dx = str(i)
            parameterdict[dx] = response_content['best parameters']
            current_usage, peak_usage = tracemalloc.get_traced_memory()
            print(f"{current_usage = }, {peak_usage = }")
            ramusagearr.append(current_usage)
            # stopping the tracemalloc library
            tracemalloc.stop()

        #Get the best parameters from the entire partitions when each of the eight models is selected
        global bestparamarr
        if model == 'pac':
            pamodelC = []
            pamodelfitintercept = []
            pamodelloss = []
            for i in range(len(dataarr)):
                pc = list(parameterdict[str(i)].values())[0]
                pamodelC.append(pc)
                pfitintercept = list(parameterdict[str(i)].values())[1]
                pamodelfitintercept.append(pfitintercept)
                ploss = list(parameterdict[str(i)].values())[2]
                pamodelloss.append(ploss)
            mode1 = stats.mode(pamodelC)[0]
            mode2 = stats.mode(pamodelfitintercept)[0]
            mode3 = stats.mode(pamodelloss)[0]
            m2 = bool(mode2[0])
            bestparamarr = {'C': mode1[0], 'fit_intercept': m2, 'loss': mode3[0]}
            print('best parameters: ', bestparamarr)

        if model == 'log':
            logregsolver = []
            logregpenalty = []
            logregC = []
            for i in range(len(dataarr)):
                lsolver = list(parameterdict[str(i)].values())[2]
                logregsolver.append(lsolver)
                lpenalty = list(parameterdict[str(i)].values())[1]
                logregpenalty.append(lpenalty)
                lC = list(parameterdict[str(i)].values())[0]
                logregC.append(lC)
            mode1 = stats.mode(logregC)[0]
            m1 = float(mode1[0])
            mode2 = stats.mode(logregpenalty)[0]
            mode3 = stats.mode(logregsolver)[0]
            # m2 = bool(mode2[0])
            bestparamarr = {'C': m1, 'penalty': mode2[0], 'solver': mode3[0]}
            print('best parameters: ', bestparamarr)

        if model == 'mnb':
            mnbalpha = []
            mnbfitprior = []
            for i in range(len(dataarr)):
                mnba = list(parameterdict[str(i)].values())[0]
                mnbalpha.append(mnba)
                mnbfitp = list(parameterdict[str(i)].values())[1]
                mnbfitprior.append(mnbfitp)
            mode1 = stats.mode(mnbalpha)[0]
            mode2 = stats.mode(mnbfitprior)[0]
            m2 = bool(mode2[0])
            bestparamarr = {'alpha': mode1[0], 'fit_prior': m2}
            print('best parameters: ', bestparamarr)

        if model == 'mlpc':
            activation = []
            hiddenlayersizes = []
            learningrate = []
            for i in range(len(dataarr)):
                act = list(parameterdict[str(i)].values())[0]
                activation.append(act)
                hidden = list(parameterdict[str(i)].values())[1]
                #print('hidden', hidden)
                hiddenarr = []
                hiddenlayerdict = {'first_hidden_layer': None, 'second_hidden_layer': None}
                hiddenlayerdict['first_hidden_layer'] = str(hidden['first_hidden_layer'])
                hiddenlayerdict['second_hidden_layer'] = str(hidden['second_hidden_layer'])

                hiddenarr.append(hidden['first_hidden_layer'])
                hiddenarr.append(hidden['second_hidden_layer'])
                hiddenlayersizes.append(hiddenarr)
                #print('hiddenlayersizes:', hiddenlayersizes)
                parameterdict[str(i)]['hidden_layer_sizes'] = hiddenlayerdict
                learnrate = list(parameterdict[str(i)].values())[2]
                learningrate.append(learnrate)
            mode1 = stats.mode(activation)[0]
            mode2 = stats.mode(hiddenlayersizes)[0]
            m2 = {'first_hidden_layer': None, 'second_hidden_layer': None}
            m2['first_hidden_layer'] = str(mode2[0][0])
            m2['second_hidden_layer'] = str(mode2[0][1])
            mode3 = stats.mode(learningrate)[0]
            m3 = float(mode3[0])
            # mode4 = stats.mode(solver)[0]
            # m2 = bool(mode2[0])
            bestparamarr = {'activation': mode1[0], 'hidden_layer_sizes': m2, 'learning_rate': m3}
            print('best parameters: ', bestparamarr)

        if model == 'svc':
            svcC = []
            svckernel = []
            svcgamma = []
            for i in range(len(dataarr)):
                sc = list(parameterdict[str(i)].values())[0]
                svcC.append(sc)
                svck = list(parameterdict[str(i)].values())[2]
                svckernel.append(svck)
                svcg = list(parameterdict[str(i)].values())[1]
                svcgamma.append(svcg)
            mode1 = stats.mode(svcC)[0]
            m1 = float(mode1[0])
            mode2 = stats.mode(svcgamma)[0]
            mode3 = stats.mode(svckernel)[0]
            # m2 = bool(mode2[0])
            bestparamarr = {'C': m1, 'gamma': mode2[0], 'kernel': mode3[0]}
            print('best parameters: ', bestparamarr)

        if model == 'dt':
            criterion = []
            max_depth = []
            max_features = []
            for i in range(len(dataarr)):
                crit = list(parameterdict[str(i)].values())[0]
                criterion.append(crit)
                md = list(parameterdict[str(i)].values())[1]
                if md == None:
                    md = 0
                else:
                    md = float(md)
                max_depth.append(md)
                mf = list(parameterdict[str(i)].values())[2]
                max_features.append(mf)
            mode1 = stats.mode(criterion)[0]
            mode2 = stats.mode(max_depth)[0]
            mode3 = stats.mode(max_features)[0]
            m1 = mode1[0]
            m2 = mode2[0]
            if m2 == 0:
                m2 = None
            m3 = mode3[0]
            # m2 = bool(mode2[0])
            bestparamarr = {'criterion': m1, 'max_depth': m2, 'max_features': m3}
            print('best parameters: ', bestparamarr)

        if model == 'kn':
            algorithm = []
            n_neighbors = []
            weights = []
            for i in range(len(dataarr)):
                algo = list(parameterdict[str(i)].values())[0]
                algorithm.append(algo)
                nneigh = list(parameterdict[str(i)].values())[1]
                nneigh = float(nneigh)
                n_neighbors.append(nneigh)
                wei = list(parameterdict[str(i)].values())[2]
                weights.append(wei)
            mode1 = stats.mode(algorithm)[0]
            mode2 = stats.mode(n_neighbors)[0]
            # m2 = float(mode2[0])
            mode3 = stats.mode(weights)[0]
            # m2 = bool(mode2[0])
            bestparamarr = {'algorithm': mode1[0], 'n_neighbors': mode2[0], 'weights': mode3[0]}
            print('best parameters: ', bestparamarr)

        if model == 'rf':
            criterion = []
            max_depth = []
            n_estimators = []
            for i in range(len(dataarr)):
                crit = list(parameterdict[str(i)].values())[0]
                criterion.append(crit)
                md = list(parameterdict[str(i)].values())[1]
                if md == None:
                    md = 0
                else:
                    md = float(md)
                max_depth.append(md)
                ne = list(parameterdict[str(i)].values())[2]
                ne = float(ne)
                n_estimators.append(ne)
            mode1 = stats.mode(criterion)[0]
            mode2 = stats.mode(max_depth)[0]
            mode3 = stats.mode(n_estimators)[0]
            m1 = mode1[0]
            m2 = mode2[0]
            if m2 == 0:
                m2 = None
            m3 = mode3[0]
            # m2 = bool(mode2[0])
            bestparamarr = {'criterion': m1, 'max_depth': m2, 'n_estimators': m3}
            print('best parameters: ', bestparamarr)

        mean_accuracy = '%.3f' % mean(accuracyarr)
        print('average validation accuracy: ', mean_accuracy)
        mean_testaccuracy = '%.3f' % mean(testaccuracyarr)
        print('average test accuracy: ', mean_testaccuracy)
        sumcurrentusage = sum(ramusagearr)
        print('memory usage', sumcurrentusage)
        print('finished')

        t1 = time.time()                                        #time at the end
        # CPU seconds elapsed (floating point)
        t1t0 = '%.3f' % (t1 - t0)
        print('time elapsed: ', '%.3f'% (t1 - t0), 'secs')
        # Get new net in/out
        net2_out = psutil.net_io_counters().bytes_sent          #bytes sent at the end
        net2_in = psutil.net_io_counters().bytes_recv           #bytes received at the end
        # Compare and get current speed
        if net1_in > net2_in:
            current_in = 0
        else:
            current_in = (net2_in - net1_in)

        if net1_out > net2_out:
            current_out = 0
        else:
            current_out = (net2_out - net1_out)
        network = {"traffic_in": current_in, "traffic_out": current_out}
        totaltraffic = current_in + current_out
        receivespeed = '%.3f' % (current_in / (t1 - t0))
        sendspeed = '%.3f' % (current_out / (t1 - t0))
        print("total traffic: ", totaltraffic, "bytes")
        print("speed of inflow traffic: ", receivespeed, "bytes/sec")
        print("speed of outflow traffic: ", sendspeed, "bytes/sec")

        # WRITING THE DATA VALUES TO A CSV FILE
        # the columns of the csv file
        columns = ['model', 'number of partitions', 'validation accuracy', 'test accuracy', 'paramater 1',
                   'parameter 2', 'parameter 3', 'time elapsed (secs)',
                   'total traffic (bytes)', 'inflow speed (bytes/sec)', 'outflow speed (bytes/sec)',
                   'memory usage (bytes)']
        # check if the csv file already exists
        if os.path.exists('results.csv') == True:
            with open('results.csv', 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                # writer.writeheader()
                if model == 'pac':
                    datapar1 = {'model': 'pac', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['C'],
                                'parameter 2': bestparamarr['fit_intercept'], 'parameter 3': bestparamarr['loss'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar1)
                if model == 'log':
                    datapar2 = {'model': 'log', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['C'],
                                'parameter 2': bestparamarr['penalty'], 'parameter 3': bestparamarr['solver'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar2)
                if model == 'mnb':
                    datapar3 = {'model': 'mnb', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['alpha'],
                                'parameter 2': bestparamarr['fit_prior'], 'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic,
                                'inflow speed (bytes/sec)': receivespeed, 'outflow speed (bytes/sec)': sendspeed,
                                'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar3)
                if model == 'mlpc':
                    hiddenlayerdata = []
                    hiddenlayerdata.append(int(bestparamarr['hidden_layer_sizes']['first_hidden_layer']))
                    hiddenlayerdata.append(int(bestparamarr['hidden_layer_sizes']['second_hidden_layer']))
                    datapar4 = {'model': 'mlpc', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['activation'],
                                'parameter 2': hiddenlayerdata, 'parameter 3': bestparamarr['learning_rate'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar4)
                if model == 'svc':
                    datapar5 = {'model': 'svc', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['C'],
                                'parameter 2': bestparamarr['gamma'], 'parameter 3': bestparamarr['kernel'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar5)
                if model == 'dt':
                    datapar6 = {'model': 'dt', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['criterion'],
                                'parameter 2': bestparamarr['max_depth'], 'parameter 3': bestparamarr['max_features'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar6)
                if model == 'kn':
                    datapar7 = {'model': 'kn', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['algorithm'],
                                'parameter 2': bestparamarr['n_neighbors'], 'parameter 3': bestparamarr['weights'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar7)
                if model == 'rf':
                    datapar8 = {'model': 'rf', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['criterion'],
                                'parameter 2': bestparamarr['max_depth'], 'parameter 3': bestparamarr['n_estimators'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar8)
        else:
            with open('results.csv', 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                if model == 'pac':
                    datapar1 = {'model': 'pac', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['C'],
                                'parameter 2': bestparamarr['fit_intercept'], 'parameter 3': bestparamarr['loss'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar1)
                if model == 'log':
                    datapar2 = {'model': 'log', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['C'],
                                'parameter 2': bestparamarr['penalty'], 'parameter 3': bestparamarr['solver'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar2)
                if model == 'mnb':
                    datapar3 = {'model': 'mnb', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['alpha'],
                                'parameter 2': bestparamarr['fit_prior'], 'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic,
                                'inflow speed (bytes/sec)': receivespeed, 'outflow speed (bytes/sec)': sendspeed,
                                'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar3)
                if model == 'mlpc':
                    hiddenlayerdata = []
                    hiddenlayerdata.append(int(bestparamarr['hidden_layer_sizes']['first_hidden_layer']))
                    hiddenlayerdata.append(int(bestparamarr['hidden_layer_sizes']['second_hidden_layer']))
                    datapar4 = {'model': 'mlpc', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['activation'],
                                'parameter 2': hiddenlayerdata, 'parameter 3': bestparamarr['learning_rate'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar4)
                if model == 'svc':
                    datapar5 = {'model': 'svc', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['C'],
                                'parameter 2': bestparamarr['gamma'], 'parameter 3': bestparamarr['kernel'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar5)
                if model == 'dt':
                    datapar6 = {'model': 'dt', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['criterion'],
                                'parameter 2': bestparamarr['max_depth'], 'parameter 3': bestparamarr['max_features'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar6)
                if model == 'kn':
                    datapar7 = {'model': 'kn', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['algorithm'],
                                'parameter 2': bestparamarr['n_neighbors'], 'parameter 3': bestparamarr['weights'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar7)
                if model == 'rf':
                    datapar8 = {'model': 'rf', 'number of partitions': n, 'validation accuracy': mean_accuracy,
                                'test accuracy': mean_testaccuracy, 'paramater 1': bestparamarr['criterion'],
                                'parameter 2': bestparamarr['max_depth'], 'parameter 3': bestparamarr['n_estimators'],
                                'time elapsed (secs)': t1t0,
                                'total traffic (bytes)': totaltraffic, 'inflow speed (bytes/sec)': receivespeed,
                                'outflow speed (bytes/sec)': sendspeed, 'memory usage (bytes)': sumcurrentusage}
                    writer.writerow(datapar8)


