from dataset_loaders import *
import evaluation
from bmvos import BMVOS
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
torch.cuda.set_device(0)


def main():

    # define model
    model = BMVOS()
    print('Network model {} loaded'.format(model.__class__.__name__))
    model.load_state_dict(torch.load('trained_model/davis_best.pth', map_location='cuda:0'))

    # define dataset
    datasets = {
        'DAVIS16_val': DAVIS_Test('../DB/DAVIS', '2016', 'val'),
        'DAVIS17_val': DAVIS_Test('../DB/DAVIS', '2017', 'val'),
        'DAVIS17_test-dev': DAVIS_Test('../DB/DAVIS', '2017', 'test-dev'),
        # 'YTVOS18_val': YTVOS_Test('../DB/YTVOS18')
    }

    # run
    for key, dataset in datasets.items():
        evaluator = evaluation.Evaluator(dataset)
        evaluator.evaluate(model, os.path.join('outputs', key))


if __name__ == '__main__':
    main()
