# import preprocessing as pp
# import pandas as pd
# from time import time

# start = time()
# data = pd.read_csv("data.csv")
# data.drop(columns = ['URL'], inplace = True)
# data = data[~data['Artist'].isna()]
# data = data[~data['Track Name'].isna()]
# data = data[data['Region'] == 'it']
# pp.encode_dates(data, inplace = True)
# print data.head()
# pp.encode_artists(data, inplace = True)
# print data.head()
# pp.encode_tracks(data, inplace = True)
# print data.head()
# print "Elapsed time: {}".format(time() - start)

import topsongs

class TestResults(object):
    '''
    This class is used to represent the results of testing a feature of the program on the same
    dataset, but with different settings.
    '''

    def __init__(self, prog, pred_type):
        '''
        The initialization method initializes a TestResults object with an associated
        program for a feature.
        It also initializes the dictionary of the settings and the dictionary of the list of
        previous settings and performances.
        The input parameter pred_type, whose the allowed values are 'classifier' and
        'regressor', defines the type of the prediction. This is used by the tester to better
        understand how to save the performances.
        '''
        self.prog = prog
        self.settings = dict()
        self.perf_history = list()
        self.pred_type = pred_type
    
    def set_setting(self, attribute, value):
        '''
        This method sets the given settings attribute to the given value.
        '''
        self.settings[attribute] = value
    
    def add_performance(self, performances):
        '''
        This method adds the given performances to the performances' history, adding to them
        the current settings.
        '''
        settings = dict(self.settings)
        for key in performances.keys():
            settings[key] = performances[key]
        self.perf_history.append(settings)
    
    def test_classifier_performance(self):
        '''
        This function executes the test with different classifiers and saves the obtained
        performance using the current settings as keys.
        '''
        from time import time
        print "Fitting a decision tree classifier..."
        self.prog.set_classifier('dtree')
        fit_start = time()
        self.prog.fit_classifier()
        fit_end = time() - fit_start
        class_start = time()
        self.prog.test_classifier()
        class_end = time() - class_start
        self.set_setting('classifier', 'dtree')
        perfs = dict()
        perfs['fit_time'] = fit_end
        perfs['class_time'] = class_end
        perfs['accuracy'] = self.prog.accuracy
        perfs['precision'] = self.prog.precision
        perfs['recall'] = self.prog.recall
        self.add_performance(perfs)
        print "Fitting a random forest classifier..."
        self.prog.set_classifier('randforest')
        fit_start = time()
        self.prog.fit_classifier()
        fit_end = time() - fit_start
        class_start = time()
        self.prog.test_classifier()
        class_end = time() - class_start
        self.set_setting('classifier', 'randforest')
        perfs = dict()
        perfs['fit_time'] = fit_end
        perfs['class_time'] = class_end
        perfs['accuracy'] = self.prog.accuracy
        perfs['precision'] = self.prog.precision
        perfs['recall'] = self.prog.recall
        self.add_performance(perfs)

def test_topsongs():
    '''
    This function is used to test the "Top Songs" feature.
    '''
    import json
    print "Computing the top 10 in Italy."
    ts = topsongs.TopSongs()
    tr = TestResults(ts, 'classifier')
    print "Considering only 'it' region..."
    ts.filter_region('it')
    tr.set_setting('region', 'it')
    tr.set_setting('split_method', 'sample')
    tr.set_setting('top_length', 10)
    for split_ratio in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
        print "Current split ratio for the training set: {}".format(split_ratio)
        ts.initialize_train_test(train_ratio=split_ratio, sample=True)
        tr.set_setting('split_ratio', split_ratio)
        tr.test_classifier_performance()
    with open("it-top10.json", "w") as fp:
        json.dump(tr.perf_history, fp, indent=4, sort_keys=True)

    print topsongs.os.linesep

    print "Computing a variable top in USA and Canada."
    ts = topsongs.TopSongs(top_length=3)
    tr = TestResults(ts, 'classifier')
    print "Considering only 'us' and 'ca' regions..."
    ts.filter_region(['us', 'ca'])
    tr.set_setting('region', ['us', 'ca'])
    ts.initialize_train_test()
    tr.set_setting('split_method', 'sample')
    tr.set_setting('split_ratio', 0.75)
    for top_length in range(3, 11):
        print "Current top length: {}".format(top_length)
        ts.top_length = top_length
        tr.set_setting('top_length', top_length)
        tr.test_classifier_performance()
    with open("usca-vartop.json", "w") as fp:
        json.dump(tr.perf_history, fp, indent=4, sort_keys=True)

    print topsongs.os.linesep

    print "Computing the Irish top 3, 4 and 5 in the first week of May based on the"
    print "Irish top 3, 4 and 5 of April."
    ts = topsongs.TopSongs(top_length=5)
    tr = TestResults(ts, 'classifier')
    print "Considering only 'ie' region..."
    ts.filter_region('ie')
    tr.set_setting('region', 'ie')
    ts.initialize_train_test(full=True)
    print "Filtering dates..."
    ts.filter_date('2017-04-01', '2017-04-30', target='train')
    ts.filter_date('2017-05-01', '2017-05-07', target='test')
    tr.set_setting('train_dates', ['2017-04-01', '2017-04-30'])
    tr.set_setting('test_dates', ['2017-05-01', '2017-05-07'])
    tr.set_setting('split_method', 'date_based')
    for top_length in range(3, 6):
        print "Current top length: {}".format(top_length)
        ts.top_length = top_length
        tr.set_setting('top_length', top_length)
        tr.test_classifier_performance()
    with open("ie-vartop-datebased.json", "w") as fp:
        json.dump(tr.perf_history, fp, indent=4, sort_keys=True)



from topsongs import TopSongs

ts = TopSongs(top_length=5, backend_predictor='randforest')
ts.filter_region('hk')
months = ['January', 'February', 'March', 'April', 'June', 'May',
          'July', 'August', 'September', 'October', 'November', 'December']
for month in range(1, 13):
    tr_lte = '2017-{:02d}-01'.format(month)
    tr_gte = '2017-{:02d}-31'.format(month)
    te_lte = '{}-{:02d}-01'.format('2017' if month < 12 else '2018', 1 if month == 12 else month + 1)
    te_gte = '{}-{:02d}-07'.format('2017' if month < 12 else '2018', 1 if month == 12 else month + 1)
    ts.initialize_train_test(full=True)
    ts.filter_date(tr_lte, tr_gte, 'train')
    ts.filter_date(te_lte, te_gte, 'test')
    ts.encode_train_and_test()
    ts.fit()
    ts.test_prediction()
    print "Predicting the first week of {} with training on the whole month of {}.".format(months[month % 12], months[month - 1])
    print "    Accuracy:   {}".format(ts.accuracy)
    print "    Precision:  {}".format(ts.precision)
    print "    Recall:     {}".format(ts.recall)