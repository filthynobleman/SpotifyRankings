# Define the list of region codes used in testing
regions = [ 'fr', 'it', 'us', 'de', 'jp', 'dk', 'pl', 'ca',
            'es', 'cz', 'ie', 'nl', 'gb', 'global']
# Define the list of region names associated to the codes in the list above
region_names = ['France', 'Italy', 'U.S.A.', 'Germany', 'Japan', 'Denmark', 'Poland',
                'Canada', 'Spain', 'Czechia', 'Ireland', 'Netherlands', 'United Kingdom',
                'Global']
# Create the dictionary that maps the region code into the region name
regnamedict = dict()
for i in range(len(regions)):
    regnamedict[regions[i]] = region_names[i]
# Create the list of metrics used in the tests
top_metrics = ['accuracy', 'precision', 'recall']
str_metrics = ['rmse', 'r2']
# Create the list of months of the years
months = ['January', 'February', 'March', 'April', 'June', 'May',
          'July', 'August', 'September', 'October', 'November', 'December']

# Import the modules to handle JSON files and dates
import json
import datetime as dt


def test_topsongs_italy_decisiontree_vs_randomforest():
    # Import the TopSongs class
    from topsongs import TopSongs
    # Create the list of classifiers and training set fractions
    classifiers = ['dtree', 'randforest']
    fractions = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    # Create the dictionary of the results
    results = dict()
    # Structure of the dictionary is
    #     classifier -> fraction -> metrics
    for classifier in classifiers:
        results[classifier] = dict()
        for val in fractions:
            results[classifier][val] = dict()
            for met in top_metrics:
                results[classifier][val][met] = None
    # Test separately the two classifiers
    for classifier in classifiers:
        print "Using backend classifier: '{}'".format(classifier)
        # Initialize the TopSongs object
        ts = TopSongs(backend_predictor=classifier)
        # Filter the dataset by region
        ts.filter_region('it')
        # For each fraction, test the predictor performances
        for val in fractions:
            # Initialize training and test sets to the current fraction
            ts.initialize_train_test(train_ratio=val)
            # Encode the training and test sets
            ts.encode_train_and_test()
            # Fit the predictor
            ts.fit()
            # Compute the prediction on the test set
            ts.test_prediction()
            print "    Performances using as training set {}% of the dataset.".format(val * 100)
            print "        Accuracy:  {}".format(ts.accuracy)
            print "        Precision: {}".format(ts.precision)
            print "        Recall:    {}".format(ts.recall)
            # Save the results
            results[classifier][val][top_metrics[0]] = ts.accuracy
            results[classifier][val][top_metrics[1]] = ts.precision
            results[classifier][val][top_metrics[2]] = ts.recall
    # Write results on disk
    with open("top10-italy-dtree-vs-randforest.json", "w") as fp:
        json.dump(results, fp)


def test_topsongs_usa_canada_decisiontree_vs_randomforest():
    # Import the TopSongs class
    from topsongs import TopSongs
    # Create the list of classifiers and training set fractions
    classifiers = ['dtree', 'randforest']
    topvalues = range(3, 11)
    # Create the dictionary of the results
    results = dict()
    # Structure of the dictionary is
    #     classifier -> top length -> metrics
    for classifier in classifiers:
        results[classifier] = dict()
        for length in topvalues:
            results[classifier][length] = dict()
            for met in top_metrics:
                results[classifier][length][met] = None
    # Test separately the two classifiers
    for classifier in classifiers:
        # For each fraction, test the predictor performances
        for length in topvalues:
            # Initialize the TopSongs object with the current top length
            ts = TopSongs(backend_predictor=classifier, top_length=length)
            # Filter the dataset by region
            ts.filter_region(['us', 'ca'])
            # Initialize training and test sets to the default value
            ts.initialize_train_test()
            # Encode the training and test sets
            ts.encode_train_and_test()
            # Fit the predictor
            ts.fit()
            # Compute the prediction on the test set
            ts.test_prediction()
            print "Performances on top {}.".format(length)
            print "    Accuracy:  {}".format(ts.accuracy)
            print "    Precision: {}".format(ts.precision)
            print "    Recall:    {}".format(ts.recall)
            # Save the results
            results[classifier][length][top_metrics[0]] = ts.accuracy
            results[classifier][length][top_metrics[1]] = ts.precision
            results[classifier][length][top_metrics[2]] = ts.recall
    # Write results on disk
    with open("vartop-usa-canada-dtree-vs-randforest.json", "w") as fp:
        json.dump(results, fp)


def test_randomforest_topsongs_regions():
    # Import the TopSongs class
    from topsongs import TopSongs
    # Create the dictionary of the results
    results = dict()
    # Structure of the dictionary is:
    #     region -> month -> metric
    # Each leaf is initialized to None
    for reg in regions:
        results[reg] = dict()
        for month in range(1, 13):
            results[reg][month] = dict()
            for met in top_metrics:
                results[reg][month][met] = None
    # Create the TopSongs object to predict the top 5 using a random forest predictor
    ts = TopSongs(backend_predictor='randforest', top_length=5)
    # Filter the dataset using only the regions defined in the list 'regions'
    ts.filter_region(regions)
    # For each region and month, predict the first week of the sequent month using the
    # data for the current month
    for reg in regions:
        print "Current region: {}.".format(regnamedict[reg])
        for month in range(1, 13):
            # Define the bound dates for the training set
            train_start = dt.date(2017, month, 1)
            try:
                train_end = dt.date(2017, month, 31)
            except:
                try:
                    train_end = dt.date(2017, month, 30)
                except:
                    train_end = dt.date(2017, month, 28)
            # Define the bound dates for the test set
            test_start = train_end + dt.timedelta(days=1)
            test_end = test_start + dt.timedelta(days=6)
            # Initialize training and test set to be copies of the dataset
            ts.initialize_train_test(full=True)
            # Filter them, dropping every row not referring to the current region
            ts.filter_region(reg, 'train')
            ts.filter_region(reg, 'test')
            # Filter them by dates
            ts.filter_date(str(train_start), str(train_end), 'train')
            ts.filter_date(str(test_start), str(test_end), 'test')
            # Encode the training and test set
            ts.encode_train_and_test()
            # Fit the classifier
            try:
                ts.fit()
            except Exception as e:
                print "    Cannot fit predictor with month {}.".format(months[month - 1])
                print e.message
                continue
            # Predict the test set
            try:
                ts.test_prediction()
            except:
                print "    Cannot predict first week of {}.".format(months[month % 12])
                continue
            print "    Prediction between {} and {} gave results:".format(test_start, test_end)
            print "        Accuracy:  {}".format(ts.accuracy)
            print "        Precision: {}".format(ts.precision)
            print "        Recall:    {}".format(ts.recall)
            # Save the results
            results[reg][month][top_metrics[0]] = ts.accuracy
            results[reg][month][top_metrics[1]] = ts.precision
            results[reg][month][top_metrics[2]] = ts.recall
    # Write results on disk
    with open("top5-randforest.json", "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    print ""
    print "Results saved on disk."
    print "\n"

    # Search for the regions who obtained the best and the worse results on June
    # in terms of precision
    maxreg = 0
    minreg = 0
    for i in range(1, len(regions)):
        if results[regions[maxreg]]["5"][top_metrics[1]] < results[regions[i]]["5"][top_metrics[1]]:
            maxreg = i
        if results[regions[minreg]]["5"][top_metrics[1]] > results[regions[i]]["5"][top_metrics[1]]:
            minreg = i


    print "Variation of the training set on the region with better precision results."
    print "Region is {}".format(regnamedict[regions[maxreg]])
    resmax = dict()
    numweek = 0
    # Try to predict the month of June using incremental training set
    test_start = dt.date(2017, 6, 1)
    test_end = dt.date(2017, 6, 7)
    train_end = dt.date(2017, 5, 31)
    train_start = train_end
    # Increment step is by two weeks
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
    # Write results in disk
    with open("top5-max-vartrain.json", "w") as fp:
        json.dump(resmax, fp, indent=4, sort_keys=True)

    # Same procedure is applied to the region with worst precision
    print "Variation of the training set on the region with lower precision results."
    print "Region is {}".format(regnamedict[regions[minreg]])
    resmin = dict()
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


def test_randomforest_todaystreams_italy():
    # Import the class TodayStreams
    from streamsprediction import TodayStreams
    # Build the results' dictionary. Dictionary's structure is:
    #     month -> metric
    results = dict()
    for month in range(1, 13):
        results[month] = dict()
        for met in str_metrics:
            results[month][met] = None
    # Initialize the TodayStreams object with predictor of type random forest
    ts = TodayStreams(backend_predictor='randforest')
    # Filter by region. Region is Italy
    ts.filter_region('it')
    # Add the column containing the number of streams in the previous day
    ts.add_previous_streams()
    # For each month, predict the first week of the next month using as training set
    # the current month
    for month in range(1, 13):
        # Define the dates bounding training and test sets
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
        # Initialize training and test set as copy of the remaining dataset
        ts.initialize_train_test(full=True)
        # Filter the training set and the test set by dates
        ts.filter_date(str(train_start), str(train_end), 'train')
        ts.filter_date(str(test_start), str(test_end), 'test')
        # Encode the training set and the test set
        ts.encode_train_and_test()
        # Fit the regressor and compute the prediction on the test set
        ts.fit()
        ts.test_prediction()
        print "Prediction between {} and {} gave results:".format(test_start, test_end)
        print "    RMSE:      {}".format(ts.rmse)
        print "    R Squared: {}".format(ts.r_squared)
        # Save the results
        results[month][str_metrics[0]] = ts.rmse
        results[month][str_metrics[1]] = ts.r_squared

    # Get the months who obtained the worst and the best RMSE
    maxmonth = 1
    minmonth = 1
    for month in range(2, 13):
        if results[month][str_metrics[0]] < results[minmonth][str_metrics[0]]:
            minmonth = month
        if results[month][str_metrics[0]] > results[maxmonth][str_metrics[0]]:
            maxmonth = month

    print "Minimum RMSE is achieved on {}.".format(months[minmonth - 1])
    print "Maximum RMSE is achieved on {}.".format(months[maxmonth - 1])
    # Write results on disk
    with open("streams-randforest.json", "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    print ""
    print "Results saved on disk."
    print "\n"

    # Try to predict the first week of January 2018 using an incremental training set
    # Incremental step is of two weeks and the maximum dimension of the training set is
    # from January 1-st 2017 to December 31-st 2017
    test_start = dt.date(2018, 1, 1)
    test_end = dt.date(2018, 1, 7)
    train_end = dt.date(2017, 12, 31)
    train_start = train_end - dt.timedelta(days=14)
    resvar = dict()
    numweeks = 2
    while train_start > dt.date(2017, 1, 1):
        print "Training set is from {} to {}.".format(train_start, train_end)
        ts.initialize_train_test(full=True)
        ts.filter_date(str(train_start), str(train_end), 'train')
        ts.filter_date(str(test_start), str(test_end), 'test')
        ts.encode_train_and_test()
        try:
            ts.fit()
            ts.test_prediction()
        except Exception as e:
            print "Something went wrong with training set {} to {}.".format(train_start, train_end)
            print e.message
            resvar[numweeks] = None
            numweeks += 2
            train_start = train_start - dt.timedelta(days=14)
            continue

        resvar[numweeks] = (ts.rmse, ts.r_squared)
        numweeks += 2
        train_start = train_start - dt.timedelta(days=14)
    # Write results on disk
    with open("streams-vartrain.json", "w") as fp:
        json.dump(resvar, fp, indent=4, sort_keys=True)


def test_todaystreams_italy_dtree_randforest_neuralnet():
    # Import TodayStreams class
    from streamsprediction import TodayStreams
    # Initialize the list of regressors
    regressors = ['dtree', 'randforest', 'neuralnet']
    # Initialize the list of fractions of the training set
    fractions = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    # Initialize the dictionary of the results
    results = dict()
    # Structure is:
    #     regressor -> fraction -> metric
    for regressor in regressors:
        results[regressor] = dict()
        for val in fractions:
            results[regressor][val] = dict()
            for met in str_metrics:
                results[regressor][val][met] = None
    # For each regressor
    for regressor in regressors:
        print "Using backend regressor '{}'".format(regressor)
        # Initialize the TodayStreams object
        ts = TodayStreams(backend_predictor=regressor)
        # Filter by region
        ts.filter_region('it')
        # Add the previous streams column
        ts.add_previous_streams()
        # For each fraction value
        for val in fractions:
            # Initialize training and test set with the current fraction
            ts.initialize_train_test(train_ratio=val)
            # Encode the training and test set
            ts.encode_train_and_test()
            # Fit the predictor
            ts.fit()
            # Compute the regression on the test set
            ts.test_prediction()
            print "    Performances using as training set {}% of the dataset.".format(val * 100)
            print "        RMSE:      {}".format(ts.rmse)
            print "        R Squared: {}".format(ts.r_squared)
            # Save the results
            results[regressor][val][str_metrics[0]] = ts.rmse
            results[regressor][val][str_metrics[1]] = ts.r_squared
    # Write results on disk
    with open("streams-italy-vartrain-dtree-randforest-nerulanet.json", "w") as fp:
        json.dump(results, fp)

def test_todaystreams_logscaled_italy_dtree_randforest_neuralnet():
    # Import TodayStreams class
    from streamsprediction import TodayStreams
    # Initialize the list of regressors
    regressors = ['dtree', 'randforest', 'neuralnet']
    # Initialize the list of fractions of the training set
    fractions = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    # Initialize the dictionary of the results
    results = dict()
    # Structure is:
    #     regressor -> fraction -> metric
    for regressor in regressors:
        results[regressor] = dict()
        for val in fractions:
            results[regressor][val] = dict()
            for met in str_metrics:
                results[regressor][val][met] = None
    # For each regressor
    for regressor in regressors:
        print "Using backend regressor '{}'".format(regressor)
        # Initialize the TodayStreams object
        ts = TodayStreams(backend_predictor=regressor)
        # Filter by region
        ts.filter_region('it')
        # Add the previous streams column
        ts.add_previous_streams()
        # Log scale the number of streams
        ts.log_scale_streams()
        # For each fraction value
        for val in fractions:
            # Initialize training and test set with the current fraction
            ts.initialize_train_test(train_ratio=val)
            # Encode the training and test set
            ts.encode_train_and_test()
            # Fit the predictor
            ts.fit()
            # Compute the regression on the test set
            ts.test_prediction()
            print "    Performances using as training set {}% of the dataset.".format(val * 100)
            print "        RMSE:      {}".format(ts.rmse)
            print "        R Squared: {}".format(ts.r_squared)
            # Save the results
            results[regressor][val][str_metrics[0]] = ts.rmse
            results[regressor][val][str_metrics[1]] = ts.r_squared
    # Write results on disk
    with open("log-streams-italy-vartrain-dtree-randforest-nerulanet.json", "w") as fp:
        json.dump(results, fp)


if __name__ == '''__main__''':
    test_todaystreams_logscaled_italy_dtree_randforest_neuralnet()