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


# In[5]:


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


# In[6]:


# various utilities
def cycle(loader):
    while True:
        for data in loader:
            yield data

def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()

def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()

def init_random(args, bs):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


# In[7]:


# Setup SGLD model and data/replay buffer
# Images generated are added to a buffer and sampled with a probability (1-\rho) for efficiency
def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
        epoch=-1 #because it needs to start at 0
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        #replay_buffer = ckpt_dict["replay_buffer"]
        replay_buffer = init_random(args, args.buffer_size)
        epoch = ckpt_dict["epoch"]

    f = f.to(device)
    return f, replay_buffer, epoch


# In[8]:


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
#        transform_train = tr.Compose(
#            [tr.ToTensor()]
#        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
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
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid,dload_test


# In[9]:


# Routine for SGLD generation of fake images
def get_sample_q(args, device):
    # setup initial data/buffers
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    # actual SGLD
    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        # here f is CCF to calculate energies
        # evaluate model, must set train back on later (TODO:but I dont need to train energies?)
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            # calculate \parial E/\partial x_{k-1}
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            # x_k = x_{k-1} + \alpha*\parial E/\partial x_{k-1} + \theta * N
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
        
        # set self.training = True
        f.train()
        
        # Returns a new Tensor, detached from the current graph
        final_samples = x_k.detach()
        
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q


# In[10]:


# calculate loss and accuracy for periodic printout
def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


# In[11]:


# save checkpoint data
def checkpoint(f, opt, epoch_no, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'epoch': epoch_no
        #"replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    #This line added so that the most recent ckpt will be saved as 'most_recent'
    t.save(ckpt_dict, os.path.join(args.save_dir,'most_recent.pt'))
    f.to(device)


# In[12]:


#Track loss for convergence
def loss_tracker(filename,save_dir,epoch,loss,correct):
    if not os.path.isfile(os.path.join(args.save_dir,filename)):
        with open(os.path.join(args.save_dir,filename),'w') as f:
            f.write("Epoch,Loss,Acc\n")
            f.write("{},{},{}\n".format(epoch,loss,correct))
    else:
        with open(os.path.join(args.save_dir,filename),'a') as f:
            f.write("{},{},{}\n".format(epoch,loss,correct))


# In[13]:


#get the newest ckpt if not using the "most_recent.pt" file
def nat_keys(word):
    def atoi(c):
        return int(c) if c.isdigit() else c
    return [atoi(c) for c in re.split('(\d+)',word)]

def get_most_recent_ckpt(dir):
    ckpt=sorted([i for i in os.listdir(dir) if 'ckpt'==i[:4]],key=nat_keys)[-1]
    return os.path.join(dir,ckpt)


# In[14]:


#This function adds or overwrites a file to the output dir named '0_readme.txt'
#That file contains what we were hoping to do with that experiment
def exp_purpose(words,filename='0_readme.txt'):
    with open(os.path.join(args.save_dir,filename),'w') as f:
        f.write(words)


# In[15]:


# main function for training
# Uses args from class below
def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    
    # store purpose of experiment
    exp_purpose("Rerun of F model, no SGLD, and with augmentation.")
    
    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    #MODEL
    sample_q = get_sample_q(args, device)
    f, replay_buffer, start_epoch = get_model_and_buffer(args, device, sample_q)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    
    f.eval()
    with t.no_grad():
        correct, loss = eval_classification(f, dload_train, device)
        print("Epoch {}: Valid Loss {}, Valid Acc {}".format(start_epoch, loss, correct))
    f.train()
    
    # loop over epochs
    for epoch in range(start_epoch+1,(args.n_epochs+start_epoch)):
        
        
        
        # decaying learning rate?
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))
            
        # loop over data in batches
        # x_p_d sample from dataset
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            # scale up lr to full over warmup time
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            #print("x_p_d_shape",x_p_d.shape)
            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            # initialize loss
            L = 0.
            
            # this maximizes log p(x) using SGLD
            if args.p_x_weight > 0:  # maximize log p(x)
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    # get data generated by SGLD
                    # In paper x_q_shape torch.Size([64, 3, 32, 32])
                    # Batch rgb 32x32
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp
                    #print("x_q_shape",x_q.shape)

                # calculate energy for training data
                fp_all = f(x_p_d)
                
                # calculate energy for SGLD generated sample
                fq_all = f(x_q)
                
                # get means
                fp = fp_all.mean()
                fq = fq_all.mean()

                # surrogate for the difference of expected value of \partial Energy/\partial x
                # and \partial Energy/\partial x
                # Need to maximize this, so preceded by minus
                l_p_x = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                   fp - fq))
                # add to loss
                L += args.p_x_weight * l_p_x

            # normal cross entropy loss function
            if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                 cur_iter,
                                                                                 l_p_y_given_x.item(),
                                                                                 acc.item()))
                # add to loss
                L += args.p_y_given_x_weight * l_p_y_given_x

            #########################
            # dont do this for paper
            #########################
            if args.p_x_y_weight > 0:  # maximize log p(x, y)
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                      fp - fq))

                # add to loss
                L += args.p_x_y_weight * l_p_x_y
            #############################
            # End dont do this for paper
            #############################

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                1/0

            # Optimize network using our loss function L
            optim.zero_grad()            
            L.backward()
            optim.step()
            cur_iter += 1

            # Plot outputs
            if cur_iter % 100 == 0:
                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        # do checkpointing
        if epoch % args.ckpt_every == 0:
            checkpoint(f, optim, epoch, f'ckpt_{epoch}.pt', args, device)

        # Print performance assesment 
        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # train set
                correct, loss = eval_classification(f, dload_train, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
                loss_tracker('train.csv',args.save_dir,epoch,loss,correct)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
                loss_tracker('test.csv',args.save_dir,epoch,loss,correct)
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                loss_tracker('validation.csv',args.save_dir,epoch,loss,correct)
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, optim, epoch, f'best_valid_ckpt.pt', args, device)    
            f.train()
            
        # do "last" checkpoint
        checkpoint(f, optim, epoch, f'ckpt_{epoch}.pt', args, device)
        
    #Added to check the checkpoint
    #return([f,optim])


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
        
        # set from inline dict
        for key in param_dict:
            #print(key, '->', param_dict[key])
            setattr(self, key, param_dict[key])


# In[17]:


# setup change from defaults
# For paper defaults
# Added best_valid_ckpt load
inline_parms = {"lr": .0001, "dataset": "cifar10", "optimizer": "adam",\
        "save_dir": './test2_X-ent_aug/',\
        "p_x_weight": 0,\
        "p_y_given_x_weight": 1.0,\
        "p_x_y_weight": 0.0,\
        "sigma": .03, "width": 10, "depth": 28, "plot_uncond": False,\
        "uncond": True,\
        "decay_epochs": [],\
        "n_epochs": 200}

# instantiate
args = train_args(inline_parms)

print("arg warmup_iters", args.warmup_iters, "lr", args.lr)

# run
main(args)


# In[ ]:




