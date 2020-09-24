import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import transforms

import os
import argparse
import nibabel as nib
import warnings

from io import BytesIO

from src import *

warnings.simplefilter("ignore", UserWarning)


parser = argparse.ArgumentParser(description='DeepMets')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='Split batch size for heavy forward (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA')
parser.add_argument('--dataset', type=str, default='/volume/vghtpe/data/METASTASES_GKaxial_all/dicom/vghtpe.csv',
                    metavar='N', help='csv or txt file that contains dicom files to be inference')
parser.add_argument('--checkpoint', type=str, default=None,
                    metavar='N', help='path of checkpoint')
parser.add_argument('--license', type=str, default=None,
                    metavar='N', help='path of DeepMets license')
parser.add_argument('--output-path', type=str, default='/volume/vghtpe/data/METASTASES_GKaxial_all/output',
                    metavar='N', help='output path name')


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


# Dataset
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

data_loader = torch.utils.data.DataLoader(
    ImageDataset(args.dataset,
                 transform=transforms.Compose([Normalize(),
                                               Split(target_key="data",
                                                     size=(1, None, None),
                                                     margin=(1, 0, 0),
                                                     feature_stride=1),
                                               ToTensor(key_type_pairs={"data": "torch.FloatTensor"}),
                                               Resize(target_keys=["data"],
                                                      to_size=(256, 256),
                                                      mode="bilinear")])),
    batch_size=1, shuffle=False, **kwargs)


# Network
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.backbone = DA_SCSE_ResNeXt50().to(device)
        self.heads = DA_SCSE_ResNeXt50_Decoder(num_classes=2, fc_classes=1).to(device)

    def forward(self, x):

        net_out = self.heads(self.backbone(x))

        return net_out


model = Model()

if args.checkpoint:
    
    if args.license:
        
        os.environ["AILABS_LICENSE_FILE"] = args.license
        
        with open(args.checkpoint, "rb") as f:
            ckpt_enc = f.read()
            
        from license import decrypt
        ckpt_dec = decrypt(ckpt_enc)

        model.load_state_dict(torch.load(BytesIO(ckpt_dec)))
    
    else:
        print("Reminder: Please contact Taiwan AI Labs if you wish to obtain the weights & license for DeepMets.")
        model.load_state_dict(torch.load(args.checkpoint))
        
else:
    print("Reminder: Please contact Taiwan AI Labs if you wish to obtain the weights & license for DeepMets.")


# Postprocess & Saving
def save(batch, net_out, target_key='x_final'):

    net_out[target_key] = F.interpolate(net_out[target_key], size=batch['data_zhw'],
                                        mode='trilinear', align_corners=True)

    net_out[target_key] = net_out[target_key].topk(1, 1, True, True)[1].squeeze().cpu().numpy()

    nib.save(nib.Nifti1Image(net_out[target_key], None), 
             os.path.join(args.output_path, batch['data_seriesID'][0] + '.nii.gz'))

    return


# Inference
def run_inference():

    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(data_loader):

            batch['data'] = batch['data'].to(device)

            split_batches, target_key = HeavyForward.batch_to_split_batches(batch, args.batch_size)

            batch.pop(target_key)

            if isinstance(split_batches, list):

                split_net_outs = []

                for split_batch in split_batches:

                    split_net_out = model.forward(split_batch)

                    split_net_outs.append(split_net_out)

                net_out = HeavyForward.concat_split_net_outs(split_net_outs)

                del split_net_outs
                torch.cuda.empty_cache()

            else:

                net_out = model.forward(split_batches)

            save(batch, net_out)

    return


# Main
def main():
    
    run_inference()


if __name__ == "__main__":

    main()