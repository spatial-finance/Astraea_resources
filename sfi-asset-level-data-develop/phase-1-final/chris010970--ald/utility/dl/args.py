import argparse

def parseTrainArguments( args=None, description='train' ):

    """
    standard command line arguments for training scripts
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_path', action="store")

    # image size
    parser.add_argument('-i', '--image_size',
                    type=int,
                    help='image size',
                    default='256')

    # epochs
    parser.add_argument('-n', '--epochs',
                    type=int,
                    help='epochs',
                    default='10')

    # batch size
    parser.add_argument('-b', '--batch_size',
                    type=int,
                    help='batch',
                    default='8')

    # model name
    parser.add_argument('-m', '--model',
                    help='model',
                    default='vgg16')

    # save path
    parser.add_argument('-s', '--save_path',
                    help='save path',
                    default=None)

    # load path
    parser.add_argument('-l', '--load_path',
                    help='load path',
                    default=None)

    # checkpoint path
    parser.add_argument('-c', '--checkpoint_path',
                    help='checkpoint path',
                    default=None)

    return parser.parse_args(args)
