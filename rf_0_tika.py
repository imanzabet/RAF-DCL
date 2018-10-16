import os
import tika

def main(args):
    import os
    if (not args.config):
        print 'reading parameters from command line ...'
        inPath = args.path
        outPath = args.output

    elif (args.config):
        print 'reading parameters from config file ...'
        from ConfigParser import SafeConfigParser

        config = SafeConfigParser()
        config.read('config_1.ini')
        jarPath = config.get('tika', 'jar_path')
        inPath = config.get('tika', 'in_path')
        outPath = config.get('tika', 'out_path')

    # cmd = 'java -jar %s -t -i %s -o %s' %(jarPath, inPath, outPath)
    # os.system(cmd)
    from tika import parser
    import codecs
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    for path, subdirs, files in os.walk(inPath):

        if files.__contains__('.DS_Store'):
            os.remove(os.path.join(path, ".DS_Store"))
            files.remove('.DS_Store')

        lenPath = len(inPath)
        for file in files:
            parsed = parser.from_file(os.path.join(path, file))
            f = codecs.open(os.path.join(outPath, file[:-4] +'.txt'), 'w',
                            encoding='utf8',
                            errors='ignore')
            f.write(parsed['content'])
            f.close()



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
                            help='The path to the original document files folder.')
        parser.add_argument('-o', '--output',
                            help='The path to the extracted text output folder')
        args = parser.parse_args()
    except:
        args = None
        print ('No arguments! using default path:')

    main(args)