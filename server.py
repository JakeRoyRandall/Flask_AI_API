from flask import Flask, request, redirect, url_for, flash, jsonify, send_file
import torch
import numpy as np
import pickle as p
import json
import os
from PIL import Image
import os.path as osp
import argparse
import numpy as np
import dnnlib
import legacy
from utilgan import latent_anima, basename, img_read
from random import random

app = Flask(__name__)

@app.route('/generate/images', methods=['GET', 'POST'])

def handleRequest():

    args = { 'out_dir': '_out', 'model': 'models/ffhq', 'size': None,
        'scale_type': 'pad', 'latmask': None, 'nXY': '1-1',
        'splitfine': 0, 'trunc': 0.8, 'digress': 0,
        'frames': 1, 'fstep': 10, 'cubic': True, 'gauss': True }

    if request.method == 'POST':
        req_json = request.get_json()
        for key in req_json.keys():
            if key in args.keys():
                args[key] = req_json[key]
            else:
                print(f'key {key} is not a valid parameter')
    img = generate(args)
    img.show()
    # img = Image.new('RGB', (250, 50), color = 'white')
    # buffer = BytesIO()
    # img.save(buffer,format="JPEG")
    # myimg = buffer.getvalue()
    # b64 = base64.b64encode(myimg).decode()
    # print(b64)
    # img_str = "data:image/png;base64," + b64
    # return f"<img src={img_str} />"
    # return send_file(img, mimetype='image/png')
    # return f"<img src='temp.jpg'/>"
    return 'cool story bro'

def generate(args):
    np.random.seed()
    device = torch.device('cpu')

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.size = args['size']
    Gs_kwargs.scale_type = args['scale_type']

    nHW = [int(s) for s in args['nXY'].split('-')][::-1]
    n_mult = nHW[0] * nHW[1]
    lmask = np.tile(np.asarray([[[[1]]]]), (1,n_mult,1,1))
    Gs_kwargs.countHW = nHW
    Gs_kwargs.splitfine = args['splitfine']
    lmask = torch.from_numpy(lmask).to(device)
    
    # load base or custom network
    pkl_name = osp.splitext(args['model'])[0]
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=False, **Gs_kwargs)['G_ema'].to(device) # type: ignore

    lats = [] # list of [frm,1,512]
    for i in range(n_mult):
        lat_tmp = latent_anima((1, Gs.z_dim), args['frames'], args['fstep'], cubic=args['cubic'], gauss=args['gauss']) # [frm,1,512]
        lats.append(lat_tmp) # list of [frm,1,512]
    latents = np.concatenate(lats, 1) # [frm,X,512]
    latents = torch.from_numpy(latents).to(device)
    
    dconst = np.zeros([1, 1, 1, 1, 1])
    dconst = torch.from_numpy(dconst).to(device)

    # labels / conditions
    label_size = Gs.c_dim
    if label_size > 0:
        labels = torch.zeros((1, n_mult, label_size), device=device) # [frm,X,lbl]
        label_ids = []
        for i in range(n_mult):
            label_ids.append(random.randint(0, label_size-1))
        for i, l in enumerate(label_ids):
            labels[:,i,l] = 1
    else:
        labels = [None]

    # generate images from latent timeline
    latent  = latents[0] # [X,512]
    label   = labels[0 % len(labels)]
    latmask = lmask[0 % len(lmask)] if lmask is not None else [None] # [X,h,w]
    dc      = dconst[0 % len(dconst)] # [X,512,4,4]

    # generate multi-latent result
    Gs = Gs.float()
    output = Gs(latent, label, force_fp32=True, truncation_psi=args['trunc'], noise_mode='const')
    output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # save image
    ext = 'png' if output.shape[3]==4 else 'jpg'
    filename = osp.join(args['out_dir'], "%06d.%s" % (0,ext))
    return Image.fromarray(output[0], 'RGB')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')