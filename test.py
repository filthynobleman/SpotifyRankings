# # import preprocessing as pp
# # import pandas as pd
# # from time import time

# # start = time()
# # data = pd.read_csv("data.csv")
# # data.drop(columns = ['URL'], inplace = True)
# # data = data[~data['Artist'].isna()]
# # data = data[~data['Track Name'].isna()]
# # data = data[data['Region'] == 'it']
# # pp.encode_dates(data, inplace = True)
# # print data.head()
# # pp.encode_artists(data, inplace = True)
# # print data.head()
# # pp.encode_tracks(data, inplace = True)
# # print data.head()
# # print "Elapsed time: {}".format(time() - start)

# import topsongs

# class TestResults(object):
#     '''
#     This class is used to represent the results of testing a feature of the program on the same
#     dataset, but with different settings.
#     '''

#     def __init__(self, prog, pred_type):
#         '''
#         The initialization method initializes a TestResults object with an associated
#         program for a feature.
#         It also initializes the dictionary of the settings and the dictionary of the list of
#         previous settings and performances.
#         The input parameter pred_type, whose the allowed values are 'classifier' and
#         'regressor', defines the type of the prediction. This is used by the tester to better
#         understand how to save the performances.
#         '''
#         self.prog = prog
#         self.settings = dict()
#         self.perf_history = list()
#         self.pred_type = pred_type
    
#     def set_setting(self, attribute, value):
#         '''
#         This method sets the given settings attribute to the given value.
#         '''
#         self.settings[attribute] = value
    
#     def add_performance(self, performances):
#         '''
#         This method adds the given performances to the performances' history, adding to them
#         the current settings.
#         '''
#         settings = dict(self.settings)
#         for key in performances.keys():
#             settings[key] = performances[key]
#         self.perf_history.append(settings)
    
#     def test_classifier_performance(self):
#         '''
#         This function executes the test with different classifiers and saves the obtained
#         performance using the current settings as keys.
#         '''
#         from time import time
#         print "Fitting a decision tree classifier..."
#         self.prog.set_classifier('dtree')
#         fit_start = time()
#         self.prog.fit_classifier()
#         fit_end = time() - fit_start
#         class_start = time()
#         self.prog.test_classifier()
#         class_end = time() - class_start
#         self.set_setting('classifier', 'dtree')
#         perfs = dict()
#         perfs['fit_time'] = fit_end
#         perfs['class_time'] = class_end
#         perfs['accuracy'] = self.prog.accuracy
#         perfs['precision'] = self.prog.precision
#         perfs['recall'] = self.prog.recall
#         self.add_performance(perfs)
#         print "Fitting a random forest classifier..."
#         self.prog.set_classifier('randforest')
#         fit_start = time()
#         self.prog.fit_classifier()
#         fit_end = time() - fit_start
#         class_start = time()
#         self.prog.test_classifier()
#         class_end = time() - class_start
#         self.set_setting('classifier', 'randforest')
#         perfs = dict()
#         perfs['fit_time'] = fit_end
#         perfs['class_time'] = class_end
#         perfs['accuracy'] = self.prog.accuracy
#         perfs['precision'] = self.prog.precision
#         perfs['recall'] = self.prog.recall
#         self.add_performance(perfs)

# def test_topsongs():
#     '''
#     This function is used to test the "Top Songs" feature.
#     '''
#     import json
#     print "Computing the top 10 in Italy."
#     ts = topsongs.TopSongs()
#     tr = TestResults(ts, 'classifier')
#     print "Considering only 'it' region..."
#     ts.filter_region('it')
#     tr.set_setting('region', 'it')
#     tr.set_setting('split_method', 'sample')
#     tr.set_setting('top_length', 10)
#     for split_ratio in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
#         print "Current split ratio for the training set: {}".format(split_ratio)
#         ts.initialize_train_test(train_ratio=split_ratio, sample=True)
#         tr.set_setting('split_ratio', split_ratio)
#         tr.test_classifier_performance()
#     with open("it-top10.json", "w") as fp:
#         json.dump(tr.perf_history, fp, indent=4, sort_keys=True)

#     print topsongs.os.linesep

#     print "Computing a variable top in USA and Canada."
#     ts = topsongs.TopSongs(top_length=3)
#     tr = TestResults(ts, 'classifier')
#     print "Considering only 'us' and 'ca' regions..."
#     ts.filter_region(['us', 'ca'])
#     tr.set_setting('region', ['us', 'ca'])
#     ts.initialize_train_test()
#     tr.set_setting('split_method', 'sample')
#     tr.set_setting('split_ratio', 0.75)
#     for top_length in range(3, 11):
#         print "Current top length: {}".format(top_length)
#         ts.top_length = top_length
#         tr.set_setting('top_length', top_length)
#         tr.test_classifier_performance()
#     with open("usca-vartop.json", "w") as fp:
#         json.dump(tr.perf_history, fp, indent=4, sort_keys=True)

#     print topsongs.os.linesep

#     print "Computing the Irish top 3, 4 and 5 in the first week of May based on the"
#     print "Irish top 3, 4 and 5 of April."
#     ts = topsongs.TopSongs(top_length=5)
#     tr = TestResults(ts, 'classifier')
#     print "Considering only 'ie' region..."
#     ts.filter_region('ie')
#     tr.set_setting('region', 'ie')
#     ts.initialize_train_test(full=True)
#     print "Filtering dates..."
#     ts.filter_date('2017-04-01', '2017-04-30', target='train')
#     ts.filter_date('2017-05-01', '2017-05-07', target='test')
#     tr.set_setting('train_dates', ['2017-04-01', '2017-04-30'])
#     tr.set_setting('test_dates', ['2017-05-01', '2017-05-07'])
#     tr.set_setting('split_method', 'date_based')
#     for top_length in range(3, 6):
#         print "Current top length: {}".format(top_length)
#         ts.top_length = top_length
#         tr.set_setting('top_length', top_length)
#         tr.test_classifier_performance()
#     with open("ie-vartop-datebased.json", "w") as fp:
#         json.dump(tr.perf_history, fp, indent=4, sort_keys=True)




import json
import datetime as dt
from topsongs import TopSongs

regions = [ 'fr', 'it', 'us', 'de', 'jp', 'dk', 'pl', 'ca',
            'es', 'cz', 'ie', 'nl', 'gb', 'global']

region_names = ['France', 'Italy', 'U.S.A.', 'Germany', 'Japan', 'Denmark', 'Poland',
                'Canada', 'Spain', 'Czechia', 'Ireland', 'Netherlands', 'United Kingdom',
                'Global']

regnamedict = dict()
for i in range(len(regions)):
    regnamedict[regions[i]] = region_names[i]

metrics = ['accuracy', 'precision', 'recall']

months = ['January', 'February', 'March', 'April', 'June', 'May',
          'July', 'August', 'September', 'October', 'November', 'December']

results = dict()
for reg in regions:
    results[reg] = dict()
    for month in range(1, 13):
        results[reg][month] = dict()
        for met in metrics:
            results[reg][month][met] = None

ts = TopSongs(backend_predictor='randforest', top_length=5)
ts.filter_region(regions)
for reg in regions:
    print "Current region: {}.".format(regnamedict[reg])
    for month in range(1, 13):
        train_start = dt.date(2017, month, 1)
        try:
            train_end = dt.date(2017, month, 31)
        except:
            try:
                train_end = dt.date(2017, month, 30)
            except:
                train_end = dt.date(2017, month, 28)
        test_start = train_end + dt.timedelta(days=1)
        test_end = test_start + dt.timedelta(days=6)
        ts.initialize_train_test(full=True)
        ts.filter_region(reg, 'train')
        ts.filter_region(reg, 'test')
        ts.filter_date(str(train_start), str(train_end), 'train')
        ts.filter_date(str(test_start), str(test_end), 'test')
        ts.encode_train_and_test()
        try:
            ts.fit()
        except Exception as e:
            print "    Cannot fit predictor with month {}.".format(months[month - 1])
            print e.message
            continue
        try:
            ts.test_prediction()
        except:
            print "    Cannot predict first week of {}.".format(months[month % 12])
            continue
        print "    Prediction between {} and {} gave results:".format(test_start, test_end)
        print "        Accuracy:  {}".format(ts.accuracy)
        print "        Precision: {}".format(ts.precision)
        print "        Recall:    {}".format(ts.recall)
        results[reg][month][metrics[0]] = ts.accuracy
        results[reg][month][metrics[1]] = ts.precision
        results[reg][month][metrics[2]] = ts.recall

with open("top5-randforest.json", "w") as fp:
    json.dump(results, fp, indent=4, sort_keys=True)

print ""
print "Results saved on disk."
print "\n"

maxreg = 0
minreg = 0
for i in range(1, len(regions)):
    if results[regions[maxreg]][5][metrics[1]] < results[regions[i]][5][metrics[1]]:
        maxreg = i
    if results[regions[minreg]][5][metrics[1]] > results[regions[i]][5][metrics[1]]:
        minreg = i


print "Variation of the training set on the region with better precision results."
print "Region is {}".format(regnamedict[regions[maxreg]])
resmax = dist()
numweek = 0
test_start = dt.date(2017, 6, 1)
test_end = dt.date(2017, 6, 7)
train_end = dt.date(2017, 5, 31)
train_start = train_end
while (train_start - dt.timedelta(days=14)).year == 2017:
    train_start = train_start - dt.timedelta(days=14)
    numweek += 2
    ts.initialize_train_test(full=True)
    ts.filter_region(regions[maxreg], 'train')
    ts.filter_region(regions[maxreg], 'test')
    ts.filter_date(str(train_start), str(train_end), 'train')
    ts.filter_date(str(test_start), str(test_end), 'test')
    ts.encode_train_and_test()
    ts.fit()
    ts.test_prediction()
    resmax[numweek] = (ts.accuracy, ts.precision, ts.recall)

with open("top5-max-vartrain.json", "w") as fp:
    json.dump(resmax, fp, indent=4, sort_keys=True)

print "Variation of the training set on the region with lower precision results."
print "Region is {}".format(regnamedict[regions[minreg]])
resmin = dist()
numweek = 0
test_start = dt.date(2017, 6, 1)
test_end = dt.date(2017, 6, 7)
train_end = dt.date(2017, 5, 31)
train_start = train_end
while (train_start - dt.timedelta(days=14)).year == 2017:
    train_start = train_start - dt.timedelta(days=14)
    numweek += 2
    ts.initialize_train_test(full=True)
    ts.filter_region(regions[minreg], 'train')
    ts.filter_region(regions[minreg], 'test')
    ts.filter_date(str(train_start), str(train_end), 'train')
    ts.filter_date(str(test_start), str(test_end), 'test')
    ts.encode_train_and_test()
    ts.fit()
    ts.test_prediction()
    resmin[numweek] = (ts.accuracy, ts.precision, ts.recall)

with open("top5-min-vartrain.json", "w") as fp:
    json.dump(resmin, fp, indent=4, sort_keys=True)


# from streamsprediction import TodayStreams

# ts = TodayStreams(backend_predictor='randforest')
# ts.filter_region('it')
# ts.add_previous_streams()
# ts.initialize_train_test()
# ts.encode_train_and_test()
# ts.fit()
# ts.test_prediction()
# print ts.rmse, ts.r_squared