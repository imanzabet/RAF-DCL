import argparse
import os
from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import pickle


def main(args):
    if (not args.config):
        print 'reading parameters from command line ...'
        rootPath = args.path
        out_path = args.output
        file_idx = args.fileIndex

    elif (args.config):
        print 'reading parameters from config file ...'
        from ConfigParser import SafeConfigParser

        config = SafeConfigParser()
        config.read('config_2.ini')
        rootPath = config.get('prediction', 'in_path')
        out_path = config.get('prediction', 'out_path')
        file_idx = config.getint('prediction', 'fileIndex')
        rfMatrix = config.get('prediction', 'rfMatrix')

    infile = os.path.join(rootPath, rfMatrix)
    # data = load_svmlight_file(infile)
    with open(infile, 'rb') as fin:
        data = pickle.load(fin)
    X_all = data['X'].tocsr()
    y_all = data['Y']
    x_test = X_all[file_idx-1]
    y_test = y_all[file_idx-1]
    rf_pkl_file = os.path.join(rootPath, 'RF_Model.pkl')
    clf = joblib.load(rf_pkl_file)
    rf_pred = clf.predict(x_test)
    rf_prob = clf.predict_proba(x_test)


    if (y_test == 1): 
        print '\nThe true class is restricted' 
    if (y_test == 0):
        print '\nThe true class is public'
    if (rf_pred == 1):
        print '\nRF prediction is restricted'
    if (rf_pred == 0):
        print '\nRF prediction is public'
    print '\nThe probabilities for different classes of the document are: %s' %(str(rf_prob[0]*100)+' %')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        help='If the parameters is extracted from config file. '
                             'If True, then the command line parameters will be bypassed. '
                             'If False, then user needs to pass parameters from command line.',
                        type=bool,
                        default=True)
    parser.add_argument('-path' ,
                        help='The path to the project folder.')
    parser.add_argument('-o'    ,   '--output',
                        help='The path as output of the project.')
    parser.add_argument('-l'    ,   '--list',
                        help='List all documents for test.',
                        type=bool,
                        default=False)
    parser.add_argument('-f'    ,   '--fileIndex',
                        help='The index number of the testing document',
                        type=int,
                        default=-1)
    args = parser.parse_args()

    main(args)
