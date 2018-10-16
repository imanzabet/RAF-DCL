import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def main(args):
    import os
    if (not args):
        inPath = '/Users/iman.zabett/rf_final/originalTxt/'
        # inPath = os.path.join(inPath)
        outPath = '/Users/iman.zabett/rf_final/originalTxt/stemmed'
    if (not args.config):
        print 'reading parameters from command line ...'
        inPath = args.path
        outPath = args.output

    elif (args.config):
        print 'reading parameters from config file ...'
        from ConfigParser import SafeConfigParser

        config = SafeConfigParser()
        config.read('config_2.ini')
        inPath = config.get('preprocess', 'in_path')
        outPath = config.get('preprocess', 'out_path')
        workDir = config.get('preprocess', 'work_dir')
        sensitive_file = config.get('preprocess', 'sensitive_file')
        num_class = config.getint('default', 'num_class')

        class_inputs = [None] * (num_class)
        in_tmp = config.get('preprocess', 'class_inputs')
        for i, indir in enumerate(in_tmp.split()):
            class_inputs[i] = indir

        class_labels = [None] * (num_class)
        lb_tmp = config.get('default', 'class_labels')
        for i, label in enumerate(lb_tmp.split()):
            class_labels[i] = label


    sensitive_file = os.path.join(workDir, sensitive_file)
    # for i in class_inputs:
    #     class_inputs[i] = os.path.join(inPath, class_inputs[i])

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # Loading sensitive file
    f = open(sensitive_file, 'r')
    sensitive_words = list()
    for line in f:
        sensitive_words.append(line.rstrip('\n'))
    f.close()

    print 'Stemming ...'
    for i, sub_path in enumerate(class_inputs):
        for path, subdirs, files in os.walk(sub_path):

            if files.__contains__('.DS_Store'):
                os.remove(os.path.join(path, ".DS_Store"))
                files.remove('.DS_Store')

            lenPath = len(sub_path)
            for file in files:
                with open(os.path.join(path, file), 'r') as f:
                    data = f.read()
                f.close()
                import re
                text = re.sub("[^a-zA-Z]", " ",  data)
                text = text.lower()
                words = text.split()

                ### Remove a set of sensitive words (words are all in lower case)
                words = [w for w in words if not w in sensitive_words]

                ### Remove stop words
                stops = set(stopwords.words("english"))
                words = [w for w in words if not w in stops]

                ### Snowball Stemmer (NLTK) for stemming words
                from nltk.stem.snowball import SnowballStemmer
                stemmer = SnowballStemmer("english")
                stemmed = []
                for word in words:
                    stemmed.append(stemmer.stem(word))
                stemmed = ' '.join(stemmed)
                sPath = outPath + '/' + class_labels[i]
                if not os.path.exists(sPath):
                    os.makedirs(sPath)
                with open(os.path.join(sPath, file), 'w') as file:
                    for s in stemmed:
                        file.write(s)
                    file.close()


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-config',
                            help='If the parameters is extracted from config file. '
                                 'If True, then the command line parameters will be bypassed. '
                                 'If False, then user needs to pass parameters from command line.',
                            type=bool,
                            default=True)
        parser.add_argument('-path',
                            help='The path to the text files folder.')
        parser.add_argument('-o', '--output',
                            help='The path to the output folder', \
                            )
        args = parser.parse_args()
    except:
        args = None
        print ('No arguments! using default path:')

    main(args)