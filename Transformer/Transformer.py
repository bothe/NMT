from __future__ import print_function

from argparse import ArgumentParser, RawTextHelpFormatter

from Train import Trainer
from type_check import bool_type_check

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

parser.add_argument('-t', '--test', type=bool_type_check, default=False,
                    help="\n테스트 모드. 테스트를 위한 checkpoint를 전달하지 않으면 error 발생\n" + "default : False\n\n")

parser.add_argument('-d', '--dataset_folder', type=str, required=True,
                    help="\nmodel의 dataset이 저장된 folder\n" +
                         "해당 argument에 대한 자세한 설명은 ReadMe.md 참고\n\n")

parser.add_argument('-r', '--result_folder', type=str, default=None,
                    help="\n모델의 진행 사항을 저장할 폴더\n" + "default : 현재 위치에 result folder 생성\n\n")

parser.add_argument('-e', '--epochs', type=int, default=50000,
                    help="\ndefault : 50000\n\n")

parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help="\ndefault : 64\n\n")

parser.add_argument('-D', '--dropout_rate', type=float, default=0.1,
                    help="\nmodel에 적용할 dropout rate (제외할 nodes의 비율을 나타냄)\n" + "default : 0.1\n\n")

parser.add_argument('-n', '--num_layers', type=int, default=6,
                    help="\nencoder와 decoder의 layer 개수\n" + "default : 6\n\n")

parser.add_argument('-s', '--embedding_size', type=int, default=512,
                    help="\nembedding layer의 size\n" + "default : 512\n\n")

parser.add_argument('-S', '--hidden_size', type=int, default=512,
                    help="\nfully connected layer의 size\n" + "default : 512\n\n")

parser.add_argument('-H', '--heads', type=int, default=8,
                    help="\n서로 다른 head의 개수\n" + "default : 8\n\n")

parser.add_argument('-c', '--use_conv', type=bool_type_check, default=False,
                    help="\nFeedForwardNetworks에서 Dense layer 대신 Conv1D layer 사용\n" + "default : False\n\n")

parser.add_argument('-l', '--label_smoothing', type=bool_type_check, default=False,
                    help="\nlabel smoothing 적용\n" + "default : False\n\n")

parser.add_argument('-p', '--ckpt_path', type=str, default=None,
                    help="\ncheckpoint path - default : None\n" +
                         "argument는 Train.py에서 folder 값 또는 checkpoint file name\n" +
                         "ex1) -c ./foo/results/2019-04-18__004330\n" +
                         "ex2) -c ./foo/results/2019-04-18__004330/ckpt.file\n\n")

parser.add_argument('-E', '--ckpt_epoch', type=int, default=None,
                    help="\ncheckpoint path가 folder일 경우 불러올 checkpoint의 epoch\n" +
                         "만약 checkpoint의 path가 folder일 때, checkpoint_epoch를 설정하지 않으면\n" +
                         "가장 최근의 checkpoint를 불러옴\n\n")

args = parser.parse_args()
kwargs = vars(args)

transformer = Trainer(**kwargs)
transformer.start()
