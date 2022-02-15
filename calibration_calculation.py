# imports
import utils # from The Google Research Authors
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import numpy as np
import wideresnet # from The Google Research Authors
import json


# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1

# images RGB 32x32
im_sz = 32
n_ch = 3


# In[4]:


# get random subset of data
class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


# In[5]:


# setup Wide_ResNet
# Uses The Google Research Authors, file wideresnet.py
class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


# In[6]:


# Energies if y=none
# EBM energy calculated as logsumexp of logits
class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            # gathers the logits along dim 1 with indeces y
            return t.gather(logits, 1, y[:, None])


# In[7]:


# various utilities
def cycle(loader):
    while True:
        for data in loader:
            yield data

def init_random(args, bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


# In[8]:


# Selected line from the function of the same name in train_wrn_ebm.py
def get_model_and_buffer(args, device):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm)
    
    print(f"loading model from {args.load_path}")
    ckpt_dict = t.load(args.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    try:
        replay_buffer = ckpt_dict["replay_buffer"]
    except:
        replay_buffer = None
    try:
        epoch = ckpt_dict["epoch"]
    except:
        epoch=0
    
    f = f.to(device)
    return f, replay_buffer, epoch


# In[18]:


# Load in chosen dataset from svhn, cifar10, cifar100
def get_data(args):
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
        transform_train = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5))]
         #lambda x: x + args.sigma * t.randn_like(x)]
    )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")

    np.random.seed(1234)

    #Set up index variables
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    train_inds = np.array(all_inds)
    train_labeled_inds = []
    other_inds = []
    #print(type(full_train),"\n-----------------------------------------------\n",train_inds)
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    
    #Assign indexes b/w train and train_labeled (Default: all are train)
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    
    #Convert to DataLoaders
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    
    return dload_train, dload_train_labeled, dload_test


# In[10]:


#To avoid repeat code and maintanence. This is for the evaluations
def eval_classification_inner(f,dload,device):
    softmax=nn.Softmax(dim=1)
    corrects, losses, logits_all = [], [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        logits_all.extend(logits)
        
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    logits_all=t.stack(logits_all)
    logits=softmax(logits_all)
    sms = logits.max(1)[0]
    cali_vals=[(a,b.item()) for a,b in zip(corrects,sms)]
    return corrects, losses, cali_vals


# In[11]:


# calculate loss and accuracy for periodic printout
def eval_classification(f, dload, device):
    corrects, losses, _ = eval_classification_inner(f,dload,device)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss, None


# In[12]:


#calculate loss and accuracy for calibration
def eval_with_calibration(f, dload, device):
    corrects, losses, cali_vals = eval_classification_inner(f,dload,device)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss, cali_vals


# In[13]:


#save the calibration data to a file
def save_calibration(filename,cali_vals):
    with open(filename,"w") as f:
        f.write("correct,softmax\n")
        for i in cali_vals:
            f.write("{},{}\n".format(i[0],i[1]))


# In[14]:


#Track loss for convergence
def loss_tracker(filename,epoch,loss,correct):
    if not os.path.isfile(os.path.join(args.save_dir,filename)):
        with open(os.path.join(args.save_dir,filename),'w') as of:
            of.write("Epoch,Loss,Acc\n")
            of.write("{},{},{}\n".format(epoch,loss,correct))
    else:
        with open(os.path.join(args.save_dir,filename),'a') as of:
            of.write("{},{},{}\n".format(epoch,loss,correct))


# In[ ]:





# In[15]:


# main function for training
# Uses args from class below
def main(args):
    ######################################################
    ###                                                ###
    ###               Closure functions                ###
    ###                                                ###
    ######################################################
    
    #Does eval and saving for the test and training datasets
    def eval_both(eval_type,with_tracker=False):
        evs=['test', 'train']
        dls=[dload_test,dload_train]
        for ev,dl in zip(evs,dls):
            correct, loss, cv = eval_type(f, dl, device)
            if eval_type==eval_with_calibration: 
                save_calibration(os.path.join(args.save_dir,f'cali_{ev}.csv'),cv)
            if with_tracker:
                loss_tracker(f'track_{ev}.csv',epoch,loss,correct)
            print(f"{ev}: Epoch {epoch}: Valid Loss {loss}, Valid Acc {correct}")
            
    ######################################################
    ###                                                ###
    ###                  Start main                    ###
    ###                                                ###
    ######################################################
            
    utils.makedirs(args.save_dir)

    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    
    # datasets - note this evaluation has no validation set
    dload_train, dload_train_labeled, dload_test = get_data(args)
    
    # select device
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print("Loading model...")
    #MODEL
    f, buffer, epoch = get_model_and_buffer(args, device)
    

    
    #Evaluation - currently set to evaluate only the train and test sets.
    f.eval()
    with t.no_grad():
        eval_both(eval_with_calibration,with_tracker=True,)
    f.train()


# In[16]:


# Setup parameters
# defaults for paper
# --lr .0001 --dataset cifar10 --optimizer adam --p_x_weight 1.0 --p_y_given_x_weight 1.0 
# --p_x_y_weight 0.0 --sigma .03 --width 10 --depth 28 --save_dir /YOUR/SAVE/DIR 
# --plot_uncond --warmup_iters 1000
#
# Regression
# {"dataset": "cifar10", "data_root": "../data", "lr": 0.0001, "decay_epochs": [160, 180], 
# "decay_rate": 0.3, "clf_only": false, "labels_per_class": -1, "optimizer": "adam", 
# "batch_size": 64, "n_epochs": 200, "warmup_iters": 1000, "p_x_weight": 1.0, 
# "p_y_given_x_weight": 1.0, "p_x_y_weight": 0.0, "dropout_rate": 0.0, "sigma": 0.03, 
# "weight_decay": 0.0, "norm": null, "n_steps": 20, "width": 10, "depth": 28, "uncond": false, 
# "class_cond_p_x_sample": false, "buffer_size": 10000, "reinit_freq": 0.05, "sgld_lr": 1.0, 
# "sgld_std": 0.01, "save_dir": "./savedir", "ckpt_every": 10, "eval_every": 1, 
# "print_every": 100, "load_path": null, "print_to_log": false, "plot_cond": false, 
#"plot_uncond": true, "n_valid": 5000, "n_classes": 10}
class train_args():
    def __init__(self, param_dict):
        # set defaults
        self.dataset = "cifar10" #, choices=["cifar10", "svhn", "cifar100"])
        self.n_classes = 100 if self.dataset == "cifar100" else 10
        self.data_root = "../data" 
        # optimization
        self.lr = 1e-4
        self.decay_epochs = [160, 180] # help="decay learning rate by decay_rate at these epochs")
        self.decay_rate = .3 # help="learning rate decay multiplier")
        self.clf_only = False #action="store_true", help="If set, then only train the classifier")
        self.labels_per_class = -1# help="number of labeled examples per class, if zero then use all labels")
        self.optimizer = "adam" #choices=["adam", "sgd"], default="adam")
        self.batch_size = 64
        self.n_epochs = 200
        self.warmup_iters = -1 # help="number of iters to linearly increase learning rate, if -1 then no warmmup")
        # loss weighting
        self.p_x_weight = 1.
        self.p_y_given_x_weight = 1.
        self.p_x_y_weight = 0.
        # regularization
        self.dropout_rate = 0.0
        self.sigma = 3e-2 # help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
        self.weight_decay = 0.0
        # network
        self.norm = None # choices=[None, "norm", "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
        # EBM specific
        self.n_steps = 20 # help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
        self.width = 10 # help="WRN width parameter")
        self.depth = 28 # help="WRN depth parameter")
        self.uncond = False # "store_true" # help="If set, then the EBM is unconditional")
        self.class_cond_p_x_sample = False #, action="store_true", help="If set we sample from p(y)p(x|y), othewise sample from p(x)," "Sample quality higher if set, but classification accuracy better if not.")
        self.buffer_size = 10000
        self.reinit_freq = .05
        self.sgld_lr = 1.0
        self.sgld_std = 1e-2
        # logging + evaluation
        self.save_dir = './experiment'
        self.ckpt_every = 10 # help="Epochs between checkpoint save")
        self.eval_every = 1 # help="Epochs between evaluation")
        self.print_every = 100 # help="Iterations between print")
        self.load_path = None # path for checkpoint to load
        self.print_to_log = False #", action="store_true", help="If true, directs std-out to log file")
        self.plot_cond = False #", action="store_true", help="If set, save class-conditional samples")
        self.plot_uncond = False #", action="store_true", help="If set, save unconditional samples")
        self.n_valid = 5000
        self.exp_reason = ""
        
        # set from inline dict
        for key in param_dict:
            #print(key, '->', param_dict[key])
            setattr(self, key, param_dict[key])


# In[23]:


#Set up changes to default params
inline_parms = {"dataset": "cifar10", "save_dir": './production/theirs5',                 "sigma": .03, "width": 10, "depth": 28, "plot_uncond": False,                 "uncond": False,                 "load_path": './train5/best_valid_ckpt.pt'}

# instantiate
args = train_args(inline_parms)

# run
main(args)

