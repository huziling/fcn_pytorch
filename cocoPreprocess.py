import os
import json


os.system('wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip') # ~19MB
os.system('unzip captions_train-val2014.zip')

val = json.load(open('./annotations/captions_val2014.json', 'r'))
train = json.load(open('./annotations/captions_train2014.json', 'r'))

# combine all images and annotations together
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

# for efficiency lets group annotations by image
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)

# create the json blob
out = []
for i, img in enumerate(imgs):
    imgid = img['id']

    # coco specific here, they store train/val images separately
    loc = 'train2014' if 'train' in img['file_name'] else 'val2014'

    jimg = {}
    jimg['file_path'] = os.path.join(loc, img['file_name'])
    jimg['id'] = imgid

    sents = []
    annotsi = itoa[imgid]
    for a in annotsi:
        sents.append(a['caption'])
    jimg['captions'] = sents
    out.append(jimg)

json.dump(out, open('coco_raw.json', 'w'))
