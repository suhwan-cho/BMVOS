from dataset import *
import evaluation
from bmvos import BMVOS
import warnings
warnings.filterwarnings('ignore')


def test(model):
    datasets = {
        'DAVIS16_val': TestDAVIS('../DB/VOS/DAVIS', '2016', 'val'),
        'DAVIS17_val': TestDAVIS('../DB/VOS/DAVIS', '2017', 'val'),
        'DAVIS17_test-dev': TestDAVIS('../DB/VOS/DAVIS', '2017', 'test-dev'),
        # 'YTVOS18_val': TestYTVOS('../DB/VOS/YTVOS18', 'val')
    }

    for key, dataset in datasets.items():
        evaluator = evaluation.Evaluator(dataset)
        evaluator.evaluate(model, os.path.join('outputs', key))


if __name__ == '__main__':

    # set device
    torch.cuda.set_device(0)

    # define model
    model = BMVOS().eval()

    # testing stage
    model.load_state_dict(torch.load('weights/BMVOS_davis.pth', map_location='cpu'))
    with torch.no_grad():
        test(model)
