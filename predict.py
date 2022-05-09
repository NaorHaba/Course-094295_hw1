import argparse

import preprocess_data
import predict_LSTM
import predict_LR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--test_directory', type=str, help='path to test csv file')
    # model parameters
    parser.add_argument('--model_type', type=str, choices=['LSTM', 'LR'], help='type of model to predict with',
                        default='LSTM')

    args = parser.parse_args()

    preprocessed_file = preprocess_data.main(args.test_directory)

    if args.model_type == 'LSTM':
        predict_LSTM.main(preprocessed_file)
    else:
        predict_LR.main(preprocessed_file)
